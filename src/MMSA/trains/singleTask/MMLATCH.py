import logging

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ...utils import MetricsTop, dict_to_str

logger = logging.getLogger('MMSA')

class MMLATCH():
    def __init__(self,args):
        self.args = args
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

    def do_train(self, model, dataloader, return_epoch_results=False):
        # TODO
        optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            factor=0.5,
            patience=self.args.patience,
            cooldown=2,
            min_lr=self.args.learning_rate / 20.0,
        )

         # initilize results
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        # Train loop
        while(True):
            # TODO
            epochs += 1
            y_pred, y_true = [], []
            losses = []
            model.train()
            train_loss = 0.0
            iter = 0
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    """
                    print(len(batch_data['raw_text']))
                    print(len(batch_data['audio_lengths']))
                    print(len(batch_data['vision_lengths']))
                    """

                    lengths = {}
                    if not self.args.need_data_aligned:
                        lengths['text'] = torch.tensor(
                            [batch_data['text'].shape[1]] * batch_data['text'].shape[0]
                            ).to(self.args.device)
                        lengths['audio'] = batch_data['audio_lengths'].to(self.args.device)
                        lengths['vision'] = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        lengths['text'] = torch.tensor(
                            [batch_data['text'].shape[1]] * batch_data['text'].shape[0]
                            ).to(self.args.device)
                        lengths['audio'] = lengths['text']
                        lengths['vision'] = lengths['text']
                    
                    iter +=1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                    # forward
                    
                    """print("Vision shape:", vision.shape)
                    print(audio[0])
                    print("Audio shape:", audio.shape)
                    print("Text shape:", text.shape)
                    print("Labels shape:",labels.shape)
                    print(labels)"""

                    outputs = model(text, audio, vision, lengths)['M']
                    # compute loss
                    loss = self.criterion(outputs, labels)
                    train_loss += loss.item()
                    y_pred.append(outputs.cpu())
                    y_true.append(labels.cpu())
                    # backward
                    loss.backward()
                    if iter % self.args.accumulation_steps == 0:
                        optimizer.step()  # type: ignore
                        optimizer.zero_grad()

                if iter % self.args.accumulation_steps == 0:
                        optimizer.step()  # type: ignore
                        # optimizer.zero_grad()

            
            train_loss = train_loss / len(dataloader['train'])
            
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] >> loss: {round(train_loss, 4)} {dict_to_str(train_results)}"
            )

            # Validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])

            # Save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)

            # Epoch results
            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)

            # Early stop    
            if epochs - best_epoch >= self.args.patience:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        # TODO
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    lengths = {}
                    if not self.args.need_data_aligned:
                        lengths['text'] = torch.tensor(
                            [batch_data['text'].shape[1]] * batch_data['text'].shape[0]
                            ).to(self.args.device)
                        lengths['audio'] = batch_data['audio_lengths'].to(self.args.device)
                        lengths['vision'] = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        lengths['text'] = torch.tensor(
                            [batch_data['text'].shape[1]] * batch_data['text'].shape[0]
                            ).to(self.args.device)
                        lengths['audio'] = lengths['text']
                        lengths['vision'] = lengths['text']

                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                    outputs = model(text, audio, vision, lengths)

                    if return_sample_results:
                        ids.extend(batch_data['id'])
                        for item in features.keys():
                            features[item].append(outputs[item].cpu().detach().numpy())
                        all_labels.extend(labels.cpu().detach().tolist())
                        preds = outputs["M"].cpu().detach().numpy()
                        # test_preds_i = np.argmax(preds, axis=1)
                        sample_results.extend(preds.squeeze())

                    loss = self.criterion(outputs['M'], labels)
                    eval_loss += loss.item()
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels
            
        return eval_results



