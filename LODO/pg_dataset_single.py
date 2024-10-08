# Importing what we need
from library_imports import *


def Graph_load_batch(name, min_num_nodes=20, max_num_nodes=17595):
    """
    load many graphs
    :return: a list of graphs
    """
    print('Loading graph dataset: ' + str(name))

    G = nx.Graph()

    # load data
    # ---------------------------------------------------------------------------------------
    print('Loading the adjacency matrix...')
    data_adj = np.loadtxt('final_step/' + name + '/A.csv', delimiter=',').astype(int)     
    # ---------------------------------------------------------------------------------------
    print('Loading the graph indicator...')
    data_graph_indicator = np.loadtxt(path + 'graph_indicator.csv', delimiter=',').astype(int)
    # ---------------------------------------------------------------------------------------
    print('Loading the graph labels...')
    data_graph_labels = np.loadtxt(path + 'graph_labels.csv', delimiter=',').astype(int)
    # ---------------------------------------------------------------------------------------
    print('Loading the node attributes...')
    data_node_att = np.loadtxt(path + 'node_attributes_raw.csv', delimiter=',')
    # ---------------------------------------------------------------------------------------
    print('Loading the node labels...')
    node_labels = np.loadtxt(path + 'output_displacement.csv', delimiter=',')
    # ---------------------------------------------------------------------------------------
    # #######################################################################################
    print('Data loaded.')
    # #######################################################################################
    # ---------------------------------------------------------------------------------------
    print('Generating data tuples...')
    data_tuple = list(map(tuple, data_adj))
    # ---------------------------------------------------------------------------------------
    print('Adding edges...')
    G.add_edges_from(data_tuple)
    # ---------------------------------------------------------------------------------------
    print('Adding features and node labels to graph nodes...')
    for i in range(node_labels.shape[0]):
        G.add_node(i + 1, feature=data_node_att[i])
        G.add_node(i + 1, label=node_labels[i])
    # ---------------------------------------------------------------------------------------
    print('Removing isolated nodes...')
    print(list(nx.isolates(G)))
    G.remove_nodes_from(list(nx.isolates(G)))
    # ---------------------------------------------------------------------------------------
    print('Splitting data into graphs...')
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator == i + 1]
        G_sub = G.subgraph(nodes)
        G_sub.graph['label'] = data_graph_labels#[i]
        if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes() <= max_num_nodes:
            graphs.append(G_sub)
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()
    # ---------------------------------------------------------------------------------------
    print('Loaded')
    return graphs

def nx_to_tg_data(graphs):
    data_list = []

    # Ensure there's at least one graph
    if len(graphs) == 0:
        return data_list

    # Handle single graph scenario
    single_graph = len(graphs) == 1

    for i in range(len(graphs)):
        graph = graphs[i].copy()

        # Relabel nodes in the graph
        mapping = {node: idx for idx, node in enumerate(graph.nodes)}
        graph = nx.relabel_nodes(graph, mapping, copy=True)

        # Extract features and labels
        feature_values = list(nx.get_node_attributes(graph, 'feature').values())
        label_values = list(nx.get_node_attributes(graph, 'label').values())
        num_nodes = len(feature_values)

        # Combine feature values into a single array
        features = np.vstack(feature_values)

        # Combine label values into a single array
        node_labels = np.vstack(label_values)

        # Reshape and convert to torch tensors
        features = features.reshape((num_nodes, 7))  # Assuming 7 features per node
        features = torch.from_numpy(features).float()

        node_labels = node_labels.reshape((num_nodes, 3))  # Assuming 3 labels per node
        node_labels = torch.from_numpy(node_labels).float()

        pos = features[:, 0:3]
        x = features[:, 3:]

        # Get edges and create edge index
        edge_index = np.array(list(graph.edges))
        edge_index = np.concatenate((edge_index, edge_index[:, ::-1]), axis=0)
        edge_index = torch.from_numpy(edge_index).long().t()

        # Create edge attributes (edge lengths)
        edge_values = []
        for edge in edge_index.t():
            node_1, node_2 = edge
            if node_1 >= len(pos) or node_2 >= len(pos):
                print(f"Invalid edge: ({node_1}, {node_2})")
                continue
            dist = torch.dist(pos[node_1], pos[node_2])
            edge_values.append([dist.item()])

        edge_values = torch.tensor(edge_values).float()

        # Print shapes for debugging
        print("Position shape:", pos.shape)
        print("Positions:", pos)
        print("Node labels shape:", node_labels.shape)
        print("Node labels:", node_labels)
        print("Feature shape:", x.shape)
        print("Features:", x)

        # Create the data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_values, pos=pos, y=node_labels)
        data_list.append(data)

        # Progress update
        if not single_graph:
            print(f"{i + 1}/{len(graphs)} data objects created.")

    if single_graph:
        print("1/1 data object created.")

    return data_list


# ----------------------------------------------------------------------------------------------------------------------
# To create pytorch geometric dataset, select the dataset_name,


dataset_name = 'final_step'



path = 'GNNs-BreastCompression/LODO/' + dataset_name + '/a/' #


graphs_all = Graph_load_batch(dataset_name)
dataset = nx_to_tg_data(graphs_all)
print('Pytorch Geometric dataset has been created.')

path_save = 'GNNs-BreastCompression/LODO/' + dataset_name + '_pickle/' +  dataset_name + '_a_raw.pickle' #

torch.save(dataset, path_save)