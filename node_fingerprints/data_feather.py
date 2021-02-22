#see reference at https://github.com/benedekrozemberczki/karateclub/blob/master/karateclub/graph_embedding/feathergraph.py
import torch
import numpy as np
from torch_geometric.utils import to_dense_adj
import networkx
import torch.nn as nn
import math
from karateclub import FeatherGraph,FeatherNode
import networkx as nx
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_points=24
def connected_component_subgraphs(G):
    return [G.subgraph(c)for c in nx.connected_components(G)]

def to_connected(cur_graph):
    sub_graphs = connected_component_subgraphs(cur_graph)
    main_graph = sub_graphs[0]
    for sg in sub_graphs:
        if len(sg.nodes()) > len(main_graph.nodes()):
            main_graph = sg
    A=nx.to_numpy_matrix(main_graph)
    return nx.convert_matrix.from_numpy_matrix(A)

def create_node_feature_matrix(graph):
    """
    Calculating the node features.

    Arg types:
        * **graph** *(NetworkX graph)* - The graph of interest.

    Return types:
        * **X** *(NumPy array)* - The node features.
    """
    log_degree = np.array([math.log(graph.degree(node) + 1) for node in range(graph.number_of_nodes())]).reshape(-1, 1)
    clustering_coefficient = np.array([nx.clustering(graph, node) for node in range(graph.number_of_nodes())]).reshape(-1, 1)
    X = np.concatenate([log_degree, clustering_coefficient], axis=1)
    return X

class MyPreTransform(object):
    def __init__(self,config):
        self.model = FeatherNode(order=5, eval_points= eval_points)
        self.model2=FeatherGraph()
        self.num_node = config.num_node
    def __call__(self, data):
        adj_mat = np.array(to_dense_adj(data.edge_index)[0])
        graph = networkx.convert_matrix.from_numpy_matrix(adj_mat)
        if not nx.is_connected(graph):
            graph=to_connected(graph)
        data.x=create_node_feature_matrix(graph)
        data.x=torch.tensor(data.x)
        #print(data.x)
        self.model.fit(graph,data.x)
        data.emb = torch.from_numpy(self.model.get_embedding()).float()
        num = self.num_node - len(data.emb)
        m = nn.ConstantPad2d((0, 0, 0, num), 0)
        mask = torch.cat((torch.ones((len(data.emb), eval_points)), torch.zeros((num, eval_points))), 0)
        data.emb = m(data.emb).float()
        data.mask = mask
        data.emb = data.emb.reshape(-1,20, eval_points)
        return data