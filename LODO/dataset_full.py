from dataset_single_formatter import *  
from dataset_formatter import *  

from adj_matrix_creator import *

# Generate training Dataset
for i in range(10): # We have 10 batches of directions  
    dataset_tr(i+1)
dataset_tr_adj_matrix_gen()

# Generate testing Dataset
for i in range(1): # One single direction
    dataset_ts(i+1)
dataset_ts_adj_matrix_gen()
