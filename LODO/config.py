from library_imports import *
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

# # device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('CUDA availability:', torch.cuda.is_available())
print(f"Using device: {device}")
writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))