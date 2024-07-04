import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.checkpoint import checkpoint
from typing import Optional, Any, cast, Callable

__all__ = ['MMLATCH']

def calc_scores(dk):
    def fn(q, k):
        return torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dk)

    return fn

def pad_mask(lengths: torch.Tensor, max_length: Optional[int] = None, device="cpu"):
    """lengths is a torch tensor"""
    if max_length is None:
        max_length = cast(int, torch.max(lengths).item())
    max_length = cast(int, max_length)
    idx = torch.arange(0, max_length).unsqueeze(0).to(device)
    mask = (idx < lengths.unsqueeze(1)).float()
    return mask

class Attention(nn.Module):
    """Some Information about Attention"""

    def __init__(
        self,
        attention_size=512,
        input_size=None,
        query_size=None,
        dropout=0.1,
        grad_checkpoint=False,
    ):
        super(Attention, self).__init__()

        if input_size is None:
            input_size = attention_size
        if query_size is None:
            query_size = input_size

        self.dk = input_size
        self.grad_checkpoint = grad_checkpoint
        self.k = nn.Linear(input_size, attention_size, bias=False)
        self.q = nn.Linear(query_size, attention_size, bias=False)
        self.v = nn.Linear(input_size, attention_size, bias=False)
        self.drop = nn.Dropout(dropout)
        self._reset_parameters()

    def forward(self, x, queries=None, values=None, attention_mask=None):
        """
        x : (B, L, D)
        queries : (B, L, D)
        values : (B, L, D)
        """
        if queries is None:
            queries = x

        if values is None:
            values = x
        k = self.k(x)  # (B, L, A)
        q = self.q(queries)  # (B, L, A)
        v = self.v(values)  # (B, L, A)

        # weights => (B, L, L)

        if self.grad_checkpoint:
            scores = checkpoint(calc_scores(self.dk), q, k)
        else:
            scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.dk)

        if attention_mask is not None:
            scores = scores + ((1 - attention_mask.unsqueeze(1)) * -1e5)
        scores = F.softmax(scores, dim=-1)
        scores = self.drop(scores)

        # out => (B, L, A)
        out = torch.bmm(scores, v)

        return out, scores

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.v.weight)


class SymmetricAttention(nn.Module):
    """Some Information about Attention"""

    def __init__(
        self,
        attention_size=512,
        input_size=None,
        dropout=0.1,
    ):
        super(SymmetricAttention, self).__init__()

        if input_size is None:
            input_size = attention_size
        self.dk = input_size
        self.kx = nn.Linear(input_size, attention_size, bias=False)
        self.qx = nn.Linear(input_size, attention_size, bias=False)
        self.vx = nn.Linear(input_size, attention_size, bias=False)
        self.ky = nn.Linear(input_size, attention_size, bias=False)
        self.qy = nn.Linear(input_size, attention_size, bias=False)
        self.vy = nn.Linear(input_size, attention_size, bias=False)
        self.drop = nn.Dropout(dropout)

        self._reset_parameters()

    def forward(self, mod1, mod2, attention_mask=None):
        """
        x : (B, L, D)
        queries : (B, L, D)
        values : (B, L, D)
        """
        k_mod1 = self.kx(mod1)
        q_mod2 = self.qy(mod2)
        v_mod1 = self.vx(mod1)

        k_mod2 = self.ky(mod2)  # (B, L, A)
        q_mod1 = self.qx(mod1)
        v_mod2 = self.vy(mod2)

        # weights => (B, L, L)

        scores_mod1 = torch.bmm(q_mod2, k_mod1.transpose(1, 2)) / math.sqrt(self.dk)
        scores_mod2 = torch.bmm(q_mod1, k_mod2.transpose(1, 2)) / math.sqrt(self.dk)

        if attention_mask is not None:
            scores_mod1 = scores_mod1 + ((1 - attention_mask.unsqueeze(1)) * -1e5)
            scores_mod2 = scores_mod2 + ((1 - attention_mask.unsqueeze(1)) * -1e5)
        scores_mod1 = F.softmax(scores_mod1, dim=-1)
        scores_mod1 = self.drop(scores_mod1)
        scores_mod2 = F.softmax(scores_mod2, dim=-1)
        scores_mod2 = self.drop(scores_mod2)

        # out => (B, L, A)
        out_mod1 = torch.bmm(scores_mod1, v_mod1)
        out_mod2 = torch.bmm(scores_mod2, v_mod2)

        # vilbert cross residual

        # v + attention(v->a)
        # a + attention(a->v)
        out_mod1 += mod2
        out_mod2 += mod1
        return out_mod1, out_mod2

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.kx.weight)
        nn.init.xavier_uniform_(self.qx.weight)
        nn.init.xavier_uniform_(self.vx.weight)
        nn.init.xavier_uniform_(self.ky.weight)
        nn.init.xavier_uniform_(self.qy.weight)
        nn.init.xavier_uniform_(self.vy.weight)

class PadPackedSequence(nn.Module):
    """Some Information about PadPackedSequence"""

    def __init__(self, batch_first=True):
        super(PadPackedSequence, self).__init__()
        self.batch_first = batch_first

    def forward(self, x, lengths):
        max_length = lengths.max().item()
        x, _ = pad_packed_sequence(
            x, batch_first=self.batch_first, total_length=max_length
        )
        return x


class PackSequence(nn.Module):
    def __init__(self, batch_first=True):
        super(PackSequence, self).__init__()
        self.batch_first = batch_first

    def forward(self, x, lengths):
        x = pack_padded_sequence(
            x, lengths, batch_first=self.batch_first, enforce_sorted=False
        )
        lengths = lengths[x.sorted_indices]
        return x, lengths


class RNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        batch_first=True,
        layers=1,
        bidirectional=False,
        merge_bi="cat",
        dropout=0,
        rnn_type="lstm",
        packed_sequence=True,
        device="cpu",
    ):

        super(RNN, self).__init__()
        self.device = device
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.merge_bi = merge_bi
        self.rnn_type = rnn_type[0].lower()

        self.out_size = hidden_size

        if bidirectional and merge_bi == "cat":
            self.out_size = 2 * hidden_size

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size,
            hidden_size,
            batch_first=batch_first,
            num_layers=layers,
            bidirectional=bidirectional,
        )
        self.drop = nn.Dropout(dropout)
        self.packed_sequence = packed_sequence

        if packed_sequence:
            self.pack = PackSequence(batch_first=batch_first)
            self.unpack = PadPackedSequence(batch_first=batch_first)

    def _merge_bi(self, forward, backward):
        if self.merge_bi == "sum":
            return forward + backward

        return torch.cat((forward, backward), dim=-1)

    def _select_last_unpadded(self, out, lengths):
        gather_dim = 1 if self.batch_first else 0
        gather_idx = (
            (lengths - 1)  # -1 to convert to indices
            .unsqueeze(1)  # (B) -> (B, 1)
            .expand((-1, self.hidden_size))  # (B, 1) -> (B, H)
            # (B, 1, H) if batch_first else (1, B, H)
            .unsqueeze(gather_dim)
        )
        # Last forward for real length or seq (unpadded tokens)
        last_out = out.gather(gather_dim, gather_idx).squeeze(gather_dim)

        return last_out

    def _final_output(self, out, lengths):
        # Collect last hidden state
        # Code adapted from https://stackoverflow.com/a/50950188

        if not self.bidirectional:
            return self._select_last_unpadded(out, lengths)

        forward, backward = (out[..., : self.hidden_size], out[..., self.hidden_size :])
        # Last backward corresponds to first token
        last_backward_out = backward[:, 0, :] if self.batch_first else backward[0, ...]
        # Last forward for real length or seq (unpadded tokens)
        last_forward_out = self._select_last_unpadded(forward, lengths)

        return self._merge_bi(last_forward_out, last_backward_out)

    def merge_hidden_bi(self, out):
        if not self.bidirectional:
            return out

        forward, backward = (out[..., : self.hidden_size], out[..., self.hidden_size :])

        return self._merge_bi(forward, backward)

    def forward(self, x, lengths, initial_hidden=None):
        self.rnn.flatten_parameters()

        if self.packed_sequence:
            lengths = lengths.to("cpu")
            x, lengths = self.pack(x, lengths)
            lengths = lengths.to(self.device)

        if initial_hidden is not None:
            out, hidden = self.rnn(x, initial_hidden)
        else:
            out, hidden = self.rnn(x)

        if self.packed_sequence:
            out = self.unpack(out, lengths)

        out = self.drop(out)
        last_timestep = self._final_output(out, lengths)
        out = self.merge_hidden_bi(out)

        return out, last_timestep, hidden


class AttentiveRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        batch_first=True,
        layers=1,
        bidirectional=False,
        merge_bi="cat",
        dropout=0.1,
        rnn_type="lstm",
        packed_sequence=True,
        attention=False,
        return_hidden=False,
        device="cpu",
    ):
        super(AttentiveRNN, self).__init__()
        self.device = device
        self.rnn = RNN(
            input_size,
            hidden_size,
            batch_first=batch_first,
            layers=layers,
            merge_bi=merge_bi,
            bidirectional=bidirectional,
            dropout=dropout,
            rnn_type=rnn_type,
            packed_sequence=packed_sequence,
            device=device,
        )
        self.out_size = self.rnn.out_size
        self.attention = None
        self.return_hidden = return_hidden

        if attention:
            self.attention = Attention(attention_size=self.out_size, dropout=dropout)

    def forward(self, x, lengths, initial_hidden=None):
        out, last_hidden, _ = self.rnn(x, lengths, initial_hidden=initial_hidden)

        if self.attention is not None:
            out, _ = self.attention(
                out, attention_mask=pad_mask(lengths, device=self.device)
            )

            if not self.return_hidden:
                out = out.sum(1)
        else:
            out = last_hidden

        return out

class FeedbackUnit(nn.Module):
    def __init__(
        self,
        hidden_dim,
        mod1_sz,
        mask_type="learnable_sequence_mask",
        dropout=0.1,
        device="cpu",
    ):
        super(FeedbackUnit, self).__init__()
        self.mask_type = mask_type
        self.mod1_sz = mod1_sz
        self.hidden_dim = hidden_dim

        if mask_type == "learnable_sequence_mask":
            self.mask1 = RNN(hidden_dim, mod1_sz, dropout=dropout, device=device)
            self.mask2 = RNN(hidden_dim, mod1_sz, dropout=dropout, device=device)
        else:
            self.mask1 = nn.Linear(hidden_dim, mod1_sz)
            self.mask2 = nn.Linear(hidden_dim, mod1_sz)

        mask_fn = {
            "learnable_static_mask": self._learnable_static_mask,
            "learnable_sequence_mask": self._learnable_sequence_mask,
        }

        self.get_mask = mask_fn[self.mask_type]

    def _learnable_sequence_mask(self, y, z, lengths=None):
        oy, _, _ = self.mask1(y, lengths)
        oz, _, _ = self.mask2(z, lengths)

        lg = (torch.sigmoid(oy) + torch.sigmoid(oz)) * 0.5

        mask = lg

        return mask

    def _learnable_static_mask(self, y, z, lengths=None):
        y = self.mask1(y)
        z = self.mask2(z)
        mask1 = torch.sigmoid(y)
        mask2 = torch.sigmoid(z)
        mask = (mask1 + mask2) * 0.5

        return mask

    def forward(self, x, y, z, lengths=None):
        mask = self.get_mask(y, z, lengths=lengths)
        mask = F.dropout(mask, p=0.2)
        x_new = x * mask

        return x_new


class Feedback(nn.Module):
    def __init__(
        self,
        hidden_dim,
        mod1_sz,
        mod2_sz,
        mod3_sz,
        mask_type="learnable_sequence_mask",
        dropout=0.1,
        device="cpu",
    ):
        super(Feedback, self).__init__()
        self.f1 = FeedbackUnit(
            hidden_dim,
            mod1_sz,
            mask_type=mask_type,
            dropout=dropout,
            device=device,
        )
        self.f2 = FeedbackUnit(
            hidden_dim,
            mod2_sz,
            mask_type=mask_type,
            dropout=dropout,
            device=device,
        )
        self.f3 = FeedbackUnit(
            hidden_dim,
            mod3_sz,
            mask_type=mask_type,
            dropout=dropout,
            device=device,
        )

    def forward(self, low_x, low_y, low_z, hi_x, hi_y, hi_z, lengths=None):
        x = self.f1(low_x, hi_y, hi_z, lengths=lengths)
        y = self.f2(low_y, hi_x, hi_z, lengths=lengths)
        z = self.f3(low_z, hi_x, hi_y, lengths=lengths)

        return x, y, z


class AttentionFuser(nn.Module):
    def __init__(self, proj_sz=None, return_hidden=True, device="cpu"):
        super(AttentionFuser, self).__init__()
        self.return_hidden = return_hidden
        self.ta = SymmetricAttention(
            attention_size=proj_sz,
            dropout=0.1,
        )

        self.va = SymmetricAttention(
            attention_size=proj_sz,
            dropout=0.1,
        )

        self.tv = SymmetricAttention(
            attention_size=proj_sz,
            dropout=0.1,
        )

        self.tav = Attention(
            attention_size=proj_sz,
            dropout=0.1,
        )

        self.out_size = 7 * proj_sz

    def forward(self, txt, au, vi):
        ta, at = self.ta(txt, au)
        va, av = self.va(vi, au)
        tv, vt = self.tv(txt, vi)

        av = va + av
        tv = vt + tv
        ta = ta + at

        tav, _ = self.tav(txt, queries=av)

        # Sum weighted attention hidden states

        if not self.return_hidden:
            txt = txt.sum(1)
            au = au.sum(1)
            vi = vi.sum(1)
            ta = ta.sum(1)
            av = av.sum(1)
            tv = tv.sum(1)
            tav = tav.sum(1)

        # B x L x 7*D
        fused = torch.cat([txt, au, vi, ta, tv, av, tav], dim=-1)

        return fused


class AttRnnFuser(nn.Module):
    def __init__(self, proj_sz=None, device="cpu", return_hidden=False):
        super(AttRnnFuser, self).__init__()
        self.att_fuser = AttentionFuser(
            proj_sz=proj_sz,
            return_hidden=True,
            device=device,
        )
        self.rnn = AttentiveRNN(
            self.att_fuser.out_size,
            proj_sz,
            bidirectional=True,
            merge_bi="cat",
            attention=True,
            device=device,
            return_hidden=return_hidden,
        )
        self.out_size = self.rnn.out_size

    def forward(self, txt, au, vi, lengths):
        att = self.att_fuser(txt, au, vi)  # B x L x 7 * D
        out = self.rnn(att, lengths)  # B x L x 2 * D

        return out


class AudioEncoder(nn.Module):
    def __init__(self, cfg, device="cpu"):
        super(AudioEncoder, self).__init__()
        self.audio = AttentiveRNN(
            cfg["input_size"],
            cfg["hidden_size"],
            batch_first=True,
            layers=cfg["layers"],
            merge_bi="sum",
            bidirectional=cfg["bidirectional"],
            dropout=cfg["dropout"],
            rnn_type=cfg["rnn_type"],
            packed_sequence=True,
            attention=cfg["attention"],
            device=device,
            return_hidden=cfg["return_hidden"],
        )
        self.out_size = self.audio.out_size

    def forward(self, x, lengths):
        x = self.audio(x, lengths)

        return x


class VisualEncoder(nn.Module):
    def __init__(self, cfg, device="cpu"):
        super(VisualEncoder, self).__init__()
        self.visual = AttentiveRNN(
            cfg["input_size"],
            cfg["hidden_size"],
            batch_first=True,
            layers=cfg["layers"],
            merge_bi="sum",
            bidirectional=cfg["bidirectional"],
            dropout=cfg["dropout"],
            rnn_type=cfg["rnn_type"],
            packed_sequence=True,
            attention=cfg["attention"],
            device=device,
            return_hidden=cfg["return_hidden"],
        )
        self.out_size = self.visual.out_size

    def forward(self, x, lengths):
        x = self.visual(x, lengths)

        return x


class GloveEncoder(nn.Module):
    def __init__(self, cfg, device="cpu"):
        super(GloveEncoder, self).__init__()
        self.text = AttentiveRNN(
            cfg["input_size"],
            cfg["hidden_size"],
            batch_first=True,
            layers=cfg["layers"],
            merge_bi="sum",
            bidirectional=cfg["bidirectional"],
            dropout=cfg["dropout"],
            rnn_type=cfg["rnn_type"],
            packed_sequence=True,
            attention=cfg["attention"],
            device=device,
            return_hidden=cfg["return_hidden"],
        )
        self.out_size = self.text.out_size

    def forward(self, x, lengths):
        x = self.text(x, lengths)

        return x


class AudioVisualTextEncoder(nn.Module):
    def __init__(
        self,
        audio_cfg=None,
        text_cfg=None,
        visual_cfg=None,
        fuse_cfg=None,
        feedback=False,
        device="cpu",
    ):
        super(AudioVisualTextEncoder, self).__init__()
        assert (
            text_cfg["attention"] and audio_cfg["attention"] and visual_cfg["attention"]
        ), "Use attention pls."

        self.feedback = feedback
        text_cfg["return_hidden"] = True
        audio_cfg["return_hidden"] = True
        visual_cfg["return_hidden"] = True

        self.text = GloveEncoder(text_cfg, device=device)
        self.audio = AudioEncoder(audio_cfg, device=device)
        self.visual = VisualEncoder(visual_cfg, device=device)

        self.fuser = AttRnnFuser(
            proj_sz=fuse_cfg["projection_size"],
            device=device,
        )

        self.out_size = self.fuser.out_size

        if feedback:
            self.fm = Feedback(
                fuse_cfg["projection_size"],
                text_cfg["input_size"],
                audio_cfg["input_size"],
                visual_cfg["input_size"],
                mask_type=fuse_cfg["feedback_type"],
                dropout=0.1,
                device=device,
            )

    def _encode(self, txt, au, vi, lengths):
        txt = self.text(txt, lengths)
        au = self.audio(au, lengths)
        vi = self.visual(vi, lengths)

        return txt, au, vi

    def _fuse(self, txt, au, vi, lengths):
        fused = self.fuser(txt, au, vi, lengths)

        return fused

    def forward(self, txt, au, vi, lengths):
        if self.feedback:
            txt1, au1, vi1 = self._encode(txt, au, vi, lengths)
            txt, au, vi = self.fm(txt, au, vi, txt1, au1, vi1, lengths=lengths)

        txt, au, vi = self._encode(txt, au, vi, lengths)
        fused = self._fuse(txt, au, vi, lengths)

        return fused


class AudioVisualTextClassifier(nn.Module):
    def __init__(
        self,
        audio_cfg=None,
        text_cfg=None,
        visual_cfg=None,
        fuse_cfg=None,
        modalities=None,
        num_classes=1,
        feedback=False,
        device="cpu",
    ):
        super(AudioVisualTextClassifier, self).__init__()
        self.modalities = modalities

        assert "text" in modalities, "No text"
        assert "audio" in modalities, "No audio"
        assert "visual" in modalities, "No visual"

        self.encoder = AudioVisualTextEncoder(
            text_cfg=text_cfg,
            audio_cfg=audio_cfg,
            visual_cfg=visual_cfg,
            fuse_cfg=fuse_cfg,
            feedback=feedback,
            device=device,
        )

        self.classifier = nn.Linear(self.encoder.out_size, num_classes)

    def forward(self, inputs):
        out = self.encoder(
            inputs["text"], inputs["audio"], inputs["visual"], inputs["lengths"]
        )

        return self.classifier(out)


class UnimodalEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        projection_size,
        layers=1,
        bidirectional=True,
        dropout=0.2,
        encoder_type="lstm",
        attention=True,
        return_hidden=False,
        device="cpu",
    ):
        super(UnimodalEncoder, self).__init__()
        self.encoder = AttentiveRNN(
            input_size,
            projection_size,
            batch_first=True,
            layers=layers,
            merge_bi="sum",
            bidirectional=bidirectional,
            dropout=dropout,
            rnn_type=encoder_type,
            packed_sequence=True,
            attention=attention,
            device=device,
            return_hidden=return_hidden,
        )
        self.out_size = self.encoder.out_size

    def forward(self, x, lengths):
        return self.encoder(x, lengths)


class AVTEncoder(nn.Module):
    def __init__(
        self,
        text_input_size,
        audio_input_size,
        visual_input_size,
        projection_size,
        text_layers=1,
        audio_layers=1,
        visual_layers=1,
        bidirectional=True,
        dropout=0.2,
        encoder_type="lstm",
        attention=True,
        feedback=False,
        feedback_type="learnable_sequence_mask",
        device="cpu",
    ):
        super(AVTEncoder, self).__init__()
        self.feedback = feedback

        self.text = UnimodalEncoder(
            text_input_size,
            projection_size,
            layers=text_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            encoder_type=encoder_type,
            attention=attention,
            return_hidden=True,
            device=device,
        )

        self.audio = UnimodalEncoder(
            audio_input_size,
            projection_size,
            layers=audio_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            encoder_type=encoder_type,
            attention=attention,
            return_hidden=True,
            device=device,
        )

        self.visual = UnimodalEncoder(
            visual_input_size,
            projection_size,
            layers=visual_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            encoder_type=encoder_type,
            attention=attention,
            return_hidden=True,
            device=device,
        )

        self.fuser = AttRnnFuser(
            proj_sz=projection_size,
            device=device,
        )

        self.out_size = self.fuser.out_size

        if feedback:
            self.fm = Feedback(
                projection_size,
                text_input_size,
                audio_input_size,
                visual_input_size,
                mask_type=feedback_type,
                dropout=0.1,
                device=device,
            )

    def _encode(self, txt, au, vi, lengths):
        txt = self.text(txt, lengths)
        au = self.audio(au, lengths)
        vi = self.visual(vi, lengths)

        return txt, au, vi

    def _fuse(self, txt, au, vi, lengths):
        fused = self.fuser(txt, au, vi, lengths)

        return fused

    def forward(self, txt, au, vi, lengths):
        if self.feedback:
            txt1, au1, vi1 = self._encode(txt, au, vi, lengths)
            txt, au, vi = self.fm(txt, au, vi, txt1, au1, vi1, lengths=lengths)

        txt, au, vi = self._encode(txt, au, vi, lengths)
        fused = self._fuse(txt, au, vi, lengths)

        return fused

# This is not used in the MMSA framework.
# The class MMLATCH below is a modified version of it.
class AVTClassifier(nn.Module):
    def __init__(
        self,
        text_input_size,
        audio_input_size,
        visual_input_size,
        projection_size,
        text_layers=1,
        audio_layers=1,
        visual_layers=1,
        bidirectional=True,
        dropout=0.2,
        encoder_type="lstm",
        attention=True,
        feedback=False,
        feedback_type="learnable_sequence_mask",
        device="cpu",
        num_classes=1,
    ):
        super(AVTClassifier, self).__init__()

        self.encoder = AVTEncoder(
            text_input_size,
            audio_input_size,
            visual_input_size,
            projection_size,
            text_layers=text_layers,
            audio_layers=audio_layers,
            visual_layers=visual_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            encoder_type=encoder_type,
            attention=attention,
            feedback=feedback,
            feedback_type=feedback_type,
            device=device,
        )

        self.classifier = nn.Linear(self.encoder.out_size, num_classes)

    def forward(self, text, audio, vision, lengths):
        out = self.encoder(
            text, audio, vision, lengths
        )

        return self.classifier(out)


class MMLATCH(nn.Module):
    def __init__(self, args):
        super(MMLATCH, self).__init__()
        self.text_input_size = args.text_input_size
        self.audio_input_size = args.audio_input_size
        self.visual_input_size = args.visual_input_size
        self.projection_size = args.projection_size
        self.text_layers = args.text_layers
        self.audio_layers = args.audio_layers
        self.visual_layers = args.visual_layers
        self.bidirectional = args.bidirectional
        self.dropout = args.dropout
        self.encoder_type = args.encoder_type
        self.attention = args.attention
        self.feedback = args.feedback
        self.feedback_type = args.feedback_type
        self.device = args.device
        self.num_classes = args.num_classes
    
        self.encoder = AVTEncoder(
            text_input_size = self.text_input_size,
            audio_input_size = self.audio_input_size,
            visual_input_size = self.visual_input_size,
            projection_size = self.projection_size,
            text_layers = self.text_layers,
            audio_layers = self.audio_layers,
            visual_layers = self.visual_layers,
            bidirectional = self.bidirectional,
            dropout = self.dropout,
            encoder_type = self.encoder_type,
            attention = self.attention,
            feedback = self.feedback,
            feedback_type = self.feedback_type,
            device = self.device
        )

        self.classifier = nn.Linear(self.encoder.out_size, self.num_classes)

    def forward(self, text, audio, vision, lengths):
        out = self.encoder(
            text, audio, vision, lengths
        )
        
        res = {
            'M': self.classifier(out)
        }

        return res