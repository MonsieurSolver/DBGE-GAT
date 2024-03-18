import torch
import networkx as nx

from tqdm import tqdm
from torch_geometric.data import Data
from utils.metrics.bpGED import bp_ged


def save_graph(graph_data):
    """
    Change the graph by naming its attributes as 'feature'

    Args:
        graph_data (Graph-NetworkX): NetworkX's graph

    Returns:
        graph (Graph-NetworkX): NetworkX's graph
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(graph_data.num_nodes))

    edges = graph_data.edge_index.t().tolist()
    graph.add_edges_from(edges)

    node_features = {j: str(graph_data.x[j].numpy()) for j in range(len(graph_data.x))}
    nx.set_node_attributes(graph, node_features, "feature")

    return graph


def dissimilarity_matrix_compute(dataset1, dataset2, ohe=False):
    """
    Calculate the DBGE matrix. Y is GED while y_real is the graph's label.

    Args:
        dataset1 (list): list of graphs
        dataset2 (list): list of graphs
        ohe (bool): True if the attributes are in one hot encoding form

    Returns:
        graphs_GED: list of graphs with the GED calculated
        graphs: list of original graphs of the dataset1
    """
    graphs_GED = []
    graphs = []

    for i in tqdm(range(len(dataset1))):

        ged_embedding = []
        graphs.append(save_graph(dataset1[i]))

        for j in range(len(dataset2)):

            if ohe:
                dataset1[i].x_ged = torch.argmax(dataset1[i].x, -1).view(-1, 1)
                dataset2[j].x_ged = torch.argmax(dataset2[j].x, -1).view(-1, 1)
                dataset1[i].x = dataset1[i].x_ged
                dataset2[j].x = dataset2[j].x_ged

            v = bp_ged(dataset1[i], dataset2[j])

            ged_embedding.append(v)

        graphs_GED.append(Data(x=dataset1[i].x,
                               edge_index=dataset1[i].edge_index,
                               y=torch.tensor(ged_embedding, dtype=torch.float),
                               y_real=dataset1[i].y)
                          )

    return graphs_GED, graphs
