from src.MMSA.run import MMSA_run, MMSA_test
from src.MMSA.config import get_config_regression
from torch import cuda
import torch

print(cuda.is_available())

# run LMF on MOSI with default hyper parameters
config = get_config_regression('mmlatch', 'mosei')
config['featurePath'] = '/home/sharing/disk3/Datasets/MMSA-Standard/MOSEI/Processed/aligned_50.pkl'
#MMSA_run('mmlatch', 'mosei', config=config, seeds=[1111], gpu_ids=[0])

for i in range(1):
    MMSA_test(config=config,
                weights_path='/home/iraklis/MMSA/saved_models/mmlatch-mosei.pth',
                feature_path='/home/sharing/disk3/Datasets/MMSA-Standard/MOSEI/Processed/aligned_50.pkl',
                idx = i)
