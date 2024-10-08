from preprocessing_dataset import *  # before preprocessing
'''
Note: We create one adjacency matrix files, for the 30 steps.

We have 30 steps and 4 directions resulting in 120 graphs per each batch of directions. 
We have 10 batches of directions therefor, we will have 1200 graphs for training/validating/testing. You create the A.csv file once, and you 
use it 10 times for dataset/a, dataset/b, ..., and dataset/j files, where dataset/a holds information about applying forces
to the prescribed load nodes for directions 1 - 4, dataset/b holds information about applying forces to the 
prescribed load nodes for directions 5 - 8, ..., and dataset/j holds information about applying forces to the
prescribed load nodes for directions 35 - 40.
'''


def dataset_tr_adj_matrix_gen():
    input_path = 'GNNs-BreastCompression/Data_Generator/input/' 

    # Initialization
    elements_filename = input_path + 'elements.csv'

    num_nodes = 17595  # number of nodes per each graph
    num_dirs = 4   # number of directions
    t_steps = 30 # number of time steps


    formatted_data_path = 'dataset/'  
    adj_matrix_filename = formatted_data_path + 'A.csv'

    A = adj_matrix_builder(elements_filename, num_nodes)
    adj_matrix_full_format(A, num_nodes, num_dirs, t_steps, adj_matrix_filename)