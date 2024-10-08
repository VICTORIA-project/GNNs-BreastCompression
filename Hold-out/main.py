from library_imports import *
from config import device, writer
from utils import *
from train import *
import matplotlib.pyplot as plt


# Specifying the dataset

dataset_name = 'dataset'

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# ----------------------------------------------------
# Load data
dataset = load_data(dataset_name)

# ----------------------------------------------------
torch.cuda.empty_cache()
# Preprocess data
dataset = data_preprocessing(dataset)

# # ----------------------------------------------------
random.Random(1).shuffle(dataset)
print('Pytorch Geometric dataset has not been shuffeled.') ###

# ----- Final Run ------


config_selected = 'config3'     #


model, learning_curve_train, learning_curve_val = train(dataset, writer, dataset_name, config_selected)