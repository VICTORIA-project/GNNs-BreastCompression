import scipy.io
import numpy as np
import pandas as pd
import csv
from csv import reader
import itertools
from numpy import genfromtxt


def adj_matrix_builder(elements, num_nodes):
    '''

    :param elements: a .csv file which contains the elements of the FE model.
    :param num_nodes: number of nodes.
    :return: the adjacency matrix.
    '''

    A = np.zeros((num_nodes, num_nodes))

    with open(elements, 'r') as read_obj:   # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)       # Iterate over each row in the csv using reader object
        for row in csv_reader:              # row variable is a list that represents a row in csv

            node_1 = int(row[0]) - 1
            node_2 = int(row[1]) - 1
            node_3 = int(row[2]) - 1
            node_4 = int(row[3]) - 1

            A[node_1, node_2] = 1
            A[node_2, node_1] = 1

            A[node_1, node_3] = 1
            A[node_3, node_1] = 1

            A[node_1, node_4] = 1
            A[node_4, node_1] = 1

            A[node_2, node_3] = 1
            A[node_3, node_2] = 1

            A[node_2, node_4] = 1
            A[node_4, node_2] = 1

            A[node_3, node_4] = 1
            A[node_4, node_3] = 1

    return A

# --------------------------------------------------------------------------------------------------

def adj_matrix_format(A, filename):
    
    '''
    :param A: an adjacency matrix
    :param filename: the name of the file where the formatted adjacency matrix will be stored.
    :return: -
    '''

    num_rows = np.size(A,0)
    num_cols = np.size(A,1)

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for i in range(num_rows):

            for j in range(num_cols):
                list = []
                if A[i,j]!=0:
                    list.append(j+1)
                    list.append(i+1)
                    writer.writerow(list)

# --------------------------------------------------------------------------------------------------

def adj_matrix_full_format(A, num_nodes, num_dir, t_steps, filename):
    '''

    :param A: an adjacency matrix
    :param filename: the name of the file where the formatted adjacency matrix will be stored.
    :return: -
    '''

    num_rows = np.size(A,0)
    num_cols = np.size(A,1)
    counter = 0

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for k in range(num_dir*t_steps):

            for i in range(num_rows):

                for j in range(num_cols):
                    list = []
                    if A[i,j]!=0:
                        list.append(j+1 + (k * num_nodes))
                        list.append(i+1 + (k * num_nodes))
                        writer.writerow(list)
                        counter = counter + 1
                        print(counter)

# --------------------------------------------------------------------------------------------------

def xyz(filename_xyz):
    '''

    :param filename_xyz: a .csv file containing the x y and z coordinates of the nodes within the FE model
    :return: a panda data frame holding the x y z coordinate information.
    '''

    data = pd.read_csv(filename_xyz, header=None)
    df = pd.DataFrame(data)
    df.columns = ['x', 'y', 'z']

    return df

# --------------------------------------------------------------------------------------------------

def node_material_assign(elements, elements_ID, num_nodes, filename):
    '''

    :param elements: csv file which contains the elements of the FE model.
    :param elements_ID: a .csv file which assigns every element to a material.
    :param num_nodes: number of nodes within the FE model.
    :param filename: a filename where every NODE will be assigned a material property (as opposed to element).
    :return: a panda data frame holding node material information.
    '''

    node_ID = np.zeros((num_nodes, 1), dtype=int)

    with open(elements, 'r') as read_obj_1:
        with open(elements_ID, 'r') as read_obj_2:

            csv_reader_1 = reader(read_obj_1) # elements
            csv_reader_2 = reader(read_obj_2) # element id

            for row_1, row_2 in itertools.zip_longest(csv_reader_1, csv_reader_2):

                val = int(row_2[0])

                for n in row_1:
                    n = int(n)-1
                    node_ID[n, 0] = val


    np.savetxt(filename, node_ID, fmt='%i')

    df = pd.DataFrame(list(node_ID), columns=['Mat_ID'])
    print(df)
    return df

# --------------------------------------------------------------------------------------------------

def node_support_assign(support_nodes, num_nodes, filename):
    '''
    :param support_nodes: a .csv file path which indicates the node ID of rigid nodes in the FE model.
    :param num_nodes: number of nodes within the FE model.
    :param filename: the place to store a column of size( num_nodes x 1) where rigid nodes have a value of 1.
    :return: a panda data frame marking rigid nodes.
    '''

    try:
        # Read the CSV file. Assuming no header as specified, so pandas assigns default integer column names
        df = pd.read_csv(support_nodes, header=None)

        # Check if the number of rows matches the number of nodes
        if len(df) == num_nodes:
            print("The number of rows is equal to the number of nodes.")
        
        # Extract the specified column by index and create a new DataFrame with a named column
        extracted_df = pd.DataFrame(df[0])  # Create a DataFrame from the series
        extracted_df.columns = ['Rigid_ID']  # Rename the column to 'Rigid_ID'

        # Save the extracted DataFrame to a new CSV file
        extracted_df.to_csv(filename, index=False)
        print(f"CSV file saved to '{filename}'.")

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return extracted_df

# --------------------------------------------------------------------------------------------------
def feature_constructor_2(num_nodes, df_xyz, df_mat_id, df_rigid_id, filename_pres_nodes,
                          filename_force_dir_x, filename_force_dir_y, filename_force_dir_z, magnitude, t_steps):
    '''
    This function is used for when we want to construct the feature matrix when we are applying prescribe forces to
    multiple nodes at the same time.

    :param num_nodes: Number of nodes within the FE mesh
    :param df_xyz: x y z data frame
    :param df_mat_id:  material id data frame
    :param df_rigid_id: support nodes data frame
    :param filename_pres_nodes:  prescribed load nodes csv file
    :param filename_force_dir:  direction of forces (normal vectors) csv file
    :param magnitude: magnitude of the force
    :param t_steps: number of time steps in the FEA
    :return: features data frame
    '''

    pres_nodes = np.genfromtxt(filename_pres_nodes, delimiter=',', dtype=int,)
    force_dir_x = genfromtxt(filename_force_dir_x, delimiter=',')
    force_dir_y = genfromtxt(filename_force_dir_y, delimiter=',')
    force_dir_z = genfromtxt(filename_force_dir_z, delimiter=',')

    # Ensure force_dir_x is a 2D array
    if force_dir_x.ndim == 1:
        force_dir_x = force_dir_x.reshape(-1, 1)
        force_dir_y = force_dir_y.reshape(-1, 1)
        force_dir_z = force_dir_z.reshape(-1, 1)
        
    idx_repeat = force_dir_x.shape[1]
    print(idx_repeat)

    df_f_1 = pd.concat([df_xyz, df_mat_id, df_rigid_id], axis=1)
    df_f_copy = pd.concat([df_xyz, df_mat_id, df_rigid_id], axis=1)

    for i in range(idx_repeat * t_steps - 1):
        frame = [df_f_1,df_f_copy]
        df_f_1 = pd.concat(frame)
        df_f_1.reset_index(drop=True, inplace=True)

    step_mag = magnitude/(t_steps * len(pres_nodes))

    force_dir_x_full = []
    force_dir_y_full = []
    force_dir_z_full = []
    force_magnitude = []
    pres_nodes_idx = []
    m = 0


    for i in range(idx_repeat):

        for j in range(t_steps):

            for k in range(len(pres_nodes)):

                force_dir_x_full.append(force_dir_x[k, i])
                force_dir_y_full.append(force_dir_y[k, i])
                force_dir_z_full.append(force_dir_z[k, i])

                updated_mag = step_mag * (j + 1)
                force_magnitude.append(updated_mag)

                idx = int(pres_nodes[k] - 1) + (num_nodes*m)   
                pres_nodes_idx.append(idx)

            m = m + 1

    force_length = t_steps * idx_repeat * num_nodes

    M = np.zeros((force_length, 4))

    # print(force_dir_x_full)
    # print(force_dir_y_full)
    # print(force_dir_z_full)
    # print(pres_nodes_idx)

    for k in range(len(pres_nodes_idx)):

        # print('------------')
        # print(pres_nodes_idx[k])
        # print(force[k, 0])
        # print(force[k, 1])
        # print(force[k, 2])
        # print(force[k, 3])
        # print('------------')

        M[pres_nodes_idx[k], 0] = force_magnitude[k]
        M[pres_nodes_idx[k], 1] = force_dir_x_full[k]
        M[pres_nodes_idx[k], 2] = force_dir_y_full[k]
        M[pres_nodes_idx[k], 3] = force_dir_z_full[k]

    # print('Printing M')
    # print(M)

    df_f_2 = pd.DataFrame(data=M, columns=["Magnitude", "F_x", "F_y", "F_z"])

    # print(df_f_1)
    # print(df_f_2)

    frame_2 = [df_f_1, df_f_2]

    df_features = pd.concat(frame_2, axis=1)

    print(df_features)

    return df_features

# --------------------------------------------------------------------------------------------------

def graph_indicator_2(num_nodes, num_dirs, t_steps):

    num_graphs =  num_dirs * t_steps

    graph_labels = []

    for i in range(num_graphs):
        for j in range(num_nodes):
            graph_labels.append(117) # We have 116 graphs in training, therefor the test graph will be 117

    df = pd.DataFrame(graph_labels)

    return df

# --------------------------------------------------------------------------------------------------

def graph_label_2(num_dirs, t_steps):

    num_graphs =  num_dirs * t_steps

    graph_labels = []

    for i in range(num_graphs):
        graph_labels.append(117) # We have 116 graphs in training, therefor the test graph will be 117

    df = pd.DataFrame(graph_labels)

    return df

# --------------------------------------------------------------------------------------------------
def output_format(filename_output, t_steps): 
    '''
    This function requires to be hardcoded.
    :param filename_output: an output file holding displacement values at x y z for t_steps
    :param t_steps: number of time steps in the FEA
    :return: a panda data frame
    '''
    output_v1 = pd.read_csv(filename_output, header=None).values
    
    start = []
    end = []

    for i in range(t_steps):

        x = 3 * (i + 1)
        start.append(x)
        end.append(x + 3)

    out_1 = output_v1[:, start[0]: end[0]]

    df_1 = pd.DataFrame(data=out_1)

    frame = [df_1] #single


    df_output = pd.concat(frame)

    df_output.reset_index(drop=True, inplace=True)

    return df_output