from library_imports import *
from utils import *
from train import *
from config import device, writer

random.seed(1)
torch.manual_seed(1)


# ------------------------------------------------------------------------------
# Specifying the dataset

dataset_name = 'dataset'
# ----------------------------------------------------
# Load data
dataset = load_data(dataset_name)

# ----------------------------------------------------
torch.cuda.empty_cache()
# Preprocess data
dataset = data_preprocessing(dataset)

# # ----------------------------------------------------
random.Random(1).shuffle(dataset)
print('Pytorch Geometric dataset has been shuffeled.')

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

# ------Final Run -----------

# If save is equal to 1, the prediction and actual values will be saved for
# further result processing.
save = 1
mean_mag_results = 1
max_error_results = 1
# ----------------------------------------------------------------------------------------------------
# Select configuration

reproduce(dataset_name, dataset, writer, 'config3', save, mean_mag_results, max_error_results) 