#see reference at https://github.com/benedekrozemberczki/karateclub/blob/master/karateclub/graph_embedding/netlsd.py
import torch
import numpy as np
from torch_geometric.utils import to_dense_adj
import networkx
import torch.nn as nn
from karateclub import NetLSD
import networkx as nx
import scipy.sparse as sps


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_points=250


def connected_component_subgraphs(G):
    return [G.subgraph(c) for c in nx.connected_components(G)]


def to_connected(cur_graph):
    sub_graphs = connected_component_subgraphs(cur_graph)
    main_graph = sub_graphs[0]
    for sg in sub_graphs:
        if len(sg.nodes()) > len(main_graph.nodes()):
            main_graph = sg
    A = nx.to_numpy_matrix(main_graph)
    return nx.convert_matrix.from_numpy_matrix(A)



class NetLSD_child(NetLSD):

    def _calculate_heat_kernel(self, eigenvalues):
        """
        Calculating the heat kernel trace of the normalized Laplacian.

        Arg types:
            * **eigenvalues** *(Numpy array)* - The eigenvalues of the graph.

        Return types:
            * **heat_kernel_trace** *(Numpy array)* - The heat kernel trace of the graph.
        """
        timescales = np.logspace(self.scale_min, self.scale_max, self.scale_steps)
        heat_kernel = []
        for idx, t in enumerate(timescales):
            heat_kernel.append(np.exp(-t * eigenvalues).tolist())
        heat_kernel = np.array(heat_kernel)
        return heat_kernel

    def _calculate_netlsd(self, graph):
        """
        Calculating the features of a graph.

        Arg types:
            * **graph** *(NetworkX graph)* - A graph to be embedded.

        Return types:
            * **hist** *(Numpy array)* - The embedding of a single graph.
        """
        graph.remove_edges_from(nx.selfloop_edges(graph))
        laplacian = sps.coo_matrix(nx.normalized_laplacian_matrix(graph, nodelist=range(graph.number_of_nodes())),
                                   dtype=np.float32)
        eigen_values = self._calculate_eigenvalues(laplacian)
        heat_kernel = self._calculate_heat_kernel(eigen_values)
        return heat_kernel

class MyPreTransform(object):
    def __init__(self,config):
        self.model = NetLSD_child()
        self.num_node = config.num_node

    def __call__(self, data):
        adj_mat = np.array(to_dense_adj(data.edge_index)[0])
        graph = networkx.convert_matrix.from_numpy_matrix(adj_mat)
        if not nx.is_connected(graph):
            graph = to_connected(graph)
        hks = self.model._calculate_netlsd(graph).transpose()
        num = self.num_node - len(hks)
        m = nn.ConstantPad2d((0, 0, 0, num), 0)
        mask = torch.cat((torch.ones((len(hks), eval_points)), torch.zeros((num, eval_points))), 0)
        hks = m(torch.from_numpy(hks)).float()[:, None, :]
        data.emb = hks
        data.mask = mask
        return data