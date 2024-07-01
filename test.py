from src.MMSA.run import MMSA_run
from torch import cuda
import torch

print(torch.version.cuda)
print(cuda.is_available())

# run LMF on MOSI with default hyper parameters
MMSA_run('lmf', 'mosei', seeds=[1111, 1112, 1113], gpu_ids=[0])