from preprocessing_single import *

# Data set 9: Applying forces to 100 nodes at a time. Forces have 30 different magnitudes and 15 different directions.

def dataset_ts(idx):

    input_path = 'GNNs-BreastCompression/Data_Generator/input/'
    output_path = 'GNNs-BreastCompression/Data_Generator/output_single/'   

    # Initialization
    elements_filename = input_path + 'elements.csv'
    element_id_filename = input_path + 'element_ID.csv'
    xyz_filename = input_path + 'xyz.csv'
    support_nodes_filename = input_path + 'bcSupportList.csv' 
    pres_nodes_filename = input_path + 'bcPrescribeList_0_based.csv' 


    num_nodes = 17595  # number of nodes per each graph
    num_p_nodes = 1129 # number of nodes on the breast surface that are not restricted
    num_dirs = 1 
    t_steps = 1 
    f_magnitude = 90  # units: Newtons

    if idx == 1:
        formatted_data_path = 'final_step/a/' 
        filename_force_dir_x = formatted_data_path + 'force_dir_x_a.csv'
        filename_force_dir_y = formatted_data_path + 'force_dir_y_a.csv'
        filename_force_dir_z = formatted_data_path + 'force_dir_z_a.csv'
    else:
        raise ValueError('Directions do not exist.')

    ######################################################################################################
    # Output file names: 1) formatted input files, 2) intermediate files, 3) formatted output files
    # ** We only need to upload the formatted input and output files to Google Drive

    # 1) formatted input files
    adj_matrix_filename = formatted_data_path + 'A.csv'
    node_att_filename = formatted_data_path + 'node_attributes.csv'
    graph_indicator_filename = formatted_data_path + 'graph_indicator.csv'
    graph_labels_filename = formatted_data_path + 'graph_labels.csv'
    # -----------------------------------------------------------------------------
    # 2) intermediate files
    node_material_id_filename = formatted_data_path + 'node_material_id.csv'
    rigid_nodes_filename = formatted_data_path + 'rigid_nodes_id.csv'
    adj_matrix_short_filename = formatted_data_path + 'A_partial.csv'
    # -----------------------------------------------------------------------------
    # 3) formatted output files
    output_displacement_filename = formatted_data_path + 'output_displacement.csv'


    ###################################################################################################################

    # Creating the feature matrix

    df_xyz = xyz(xyz_filename)
    df_mat_id = node_material_assign(elements_filename, element_id_filename, num_nodes, node_material_id_filename)

    df_rigid_id = node_support_assign(support_nodes_filename, num_nodes, rigid_nodes_filename)

    df_features = feature_constructor_2(num_nodes, df_xyz, df_mat_id, df_rigid_id, pres_nodes_filename,
                              filename_force_dir_x, filename_force_dir_y, filename_force_dir_z, f_magnitude, t_steps)

    df_features.to_csv(node_att_filename, encoding='utf-8', header=False, index=False)

    ###################################################################################################################

    # Creating the Graph indicator file (which determines which node belongs to which graph)

    node_graph_labels = graph_indicator_2(num_nodes, num_dirs, t_steps)
    node_graph_labels.to_csv(graph_indicator_filename, header=False, index=False)

    # ###################################################################################################################
    #
    # # Graph labels

    graph_labels = graph_label_2(num_dirs, t_steps)
    graph_labels.to_csv(graph_labels_filename, header=False, index=False)

    # ###################################################################################################################
    #
    # Output for 1129 load nodes, 1 directions, and time step no. 30. Force is applied to all nodes at once.

    output_files = []

    
    for j in range(1, t_steps+1):
        s1 = "disp_"
        s4 = str(j)
        s5 = ".csv"
        s = output_path + s1 + s4 + s5
        output_files.append(s)
        print(s)

    dfs = []

    # Loop through each output file
    for output_file in output_files:
        # Print the filename
        print(f"Filename: {output_file}")
        
        # Repeat the code block 30 times for each output file
        for _ in range(num_dirs):
            # Read the CSV file without header
            df = pd.read_csv(output_file, header=None)
            
            # Append the dataframe to the list
            dfs.append(df)

    # Concatenate the dataframes
    df_output = pd.concat([pd.DataFrame(df) for df in dfs], ignore_index=True)

    df_output.to_csv(output_displacement_filename, header=False, index=False)

    print(df_output)

    # ##################################################################################################################
    # ##################################################################################################################
    # ##################################################################################################################
    # ##################################################################################################################

    # FEATURE PREPROCESSING

    data_node_att = np.loadtxt(formatted_data_path + 'node_attributes.csv', delimiter=',')
    # --------------------------------------------------------------------------------------------------------------------
    # Node attribute format
    # {x} {y} {z} {material_ID} {Rigid} {Force magnitude} {F_x} {F_y} {F_z}
    #  0   1   2       3           4            5           6     7     8
    # material id 1: brain, 2: tumour
    # ----------------------------------------------------------------------------------------------------------------------

    # ############# Continuous encoding of rigid ID and material ID ###############
    physics_prop = np.zeros((data_node_att.shape[0], 1), dtype=float)

    for j in range(data_node_att.shape[0]):
        if data_node_att[j, 4] == 1:
            physics_prop[j, 0] = 0
        else:
            if data_node_att[j, 3] == 1: # if the node belongs to fat give it 1
                physics_prop[j, 0] = 1.0
            elif data_node_att[j, 3] == 2: # if the node belongs to gland give it 0.6
                physics_prop[j, 0] = 0.6
            else:
                physics_prop[j, 0] = 0.1  # if the node belong to skin give it 0.1
    # ############################################################################
    # ############# Multiplication of force magnitude by direction ###############
    force_magnitude = data_node_att[:, 5]
    x_mag_and_direction = np.multiply(data_node_att[:, 6], force_magnitude)
    y_mag_and_direction = np.multiply(data_node_att[:, 7], force_magnitude)
    z_mag_and_direction = np.multiply(data_node_att[:, 8], force_magnitude)
    # ----------------------------------------------------------------------------------------------------------------------
    x_mag_and_direction = np.reshape(x_mag_and_direction, (-1, 1))
    y_mag_and_direction = np.reshape(y_mag_and_direction, (-1, 1))
    z_mag_and_direction = np.reshape(z_mag_and_direction, (-1, 1))

    # ----------------------------------------------------------------------------------------------------------------------
    feature_normalized = np.concatenate((data_node_att[:, 0:3], physics_prop, x_mag_and_direction, y_mag_and_direction,
                                         z_mag_and_direction), axis=1)
    # ----------------------------------------------------------------------------------------------------------------------
    print(feature_normalized)

    print(feature_normalized.shape)

    r = np.ptp(feature_normalized, axis=0)

    print(r)

    np.savetxt(formatted_data_path + "/node_attributes_raw.csv", feature_normalized, delimiter=",",
               fmt=('%f, %f, %f, %f, %f, %f, %f'))