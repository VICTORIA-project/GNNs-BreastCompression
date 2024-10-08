from library_imports import *
from utils import *
from train_single import *
from config import device, writer

random.seed(1)
torch.manual_seed(1)


# ------------------------------------------------------------------------------
# Specifying the dataset

dataset_name = 'dataset'

test_set = 'final_step'

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

reproduce(dataset_name, dataset, final_test, writer, 'config3', save, mean_mag_results, max_error_results) 