from library_imports import *
from config import device, writer
from utils import *
from train_single import *
import matplotlib.pyplot as plt


# Specifying the dataset

dataset_name = 'dataset'

test_set = 'final_step'

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# ----------------------------------------------------
# Load data
dataset = load_data(dataset_name)

# Load final test
final_test = load_data_test(test_set)

# ----------------------------------------------------
torch.cuda.empty_cache()
# Preprocess data
dataset, mean_val, std_val = data_preprocessing(dataset)

# Preprocess Test data
final_test = data_preprocessing_test(final_test, mean_val, std_val)

# # ----------------------------------------------------

print('Pytorch Geometric dataset has not been shuffeled.') ###

# ----- Final Run ------


config_selected = 'config3'     #


model, learning_curve_train, learning_curve_val = train(dataset, final_test, writer, dataset_name, config_selected)