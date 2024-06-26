import logging

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from ...utils import MetricsTop, dict_to_str

logger = logging.getLogger('MMLATCH')

class MMLATCH():
    def __init__(self,args):
        self.args = args
        pass

    def do_train(self, model, dataloader, return_epoch_results=False):
        # TODO
        pass

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        # TODO
        pass


