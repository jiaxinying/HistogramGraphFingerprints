#See reference at https://github.com/benedekrozemberczki/karateclub/blob/master/karateclub/graph_embedding/geoscattering.py
import torch
from torch_geometric.utils import to_dense_adj
import networkx
import numpy as np
import networkx as nx
from karateclub import GeoScattering
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_points=74


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



class GeoScattering_child(GeoScattering):
    def _create_node_feature_matrix(self, graph):
        """
        Calculating the node features.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph of interest.

        Return types:
            * **X** *(NumPy array)* - The node features.
        """
        log_degree = np.array([math.log(graph.degree(node) + 1) for node in range(graph.number_of_nodes())]).reshape(-1,
                                                                                                                     1)
        eccentricity = np.array([nx.eccentricity(graph, node) for node in range(graph.number_of_nodes())]).reshape(-1,
                                                                                                                   1)
        X = np.concatenate([log_degree, eccentricity], axis=1)
        return X

    def _get_zero_order_features(self, X):
        """
        Calculating the zero-th order graph features.

        Arg types:
            * **X** *(NumPy array)* - The node features.

        Return types:
            * **features** *(NumPy vector)* - The zero-th order graph features.
        """
        features = []
        X = np.abs(X)
        for col in range(X.shape[1]):
            x = np.abs(X[:, col])
            for power in range(1, self.order+1):
                features.append(np.power(x, power))
        features = np.array(features)
        return features


    def _get_first_order_features(self, Psi, X):
        """
        Calculating the first order graph features.

        Arg types:
            * **Psi** *(List of Scipy arrays)* - The wavelet matrices.
            * **X** *(NumPy array)* - The node features.

        Return types:
            * **features** *(NumPy vector)* - The first order graph features.
        """
        features = []
        X = np.abs(X)
        for col in range(X.shape[1]):
            x = np.abs(X[:, col])
            for psi in Psi:
                filtered_x = psi.dot(x)
                for q in range(1, self.moments):
                    features.append(np.power(np.abs(filtered_x), q))

        features = np.array(features)
        return features

    def _get_second_order_features(self, Psi, X):
        """
        Calculating the second order graph features.

        Arg types:
            * **Psi** *(List of Scipy arrays)* - The wavelet matrices.
            * **X** *(NumPy array)* - The node features.

        Return types:
            * **features** *(NumPy vector)* - The second order graph features.
        """
        features = []
        X = np.abs(X)
        for col in range(X.shape[1]):
            x = np.abs(X[:, col])
            for i in range(self.order - 1):
                for j in range(i + 1, self.order):
                    psi_j = Psi[i]
                    psi_j_prime = Psi[j]
                    filtered_x = np.abs(psi_j_prime.dot(np.abs(psi_j.dot(x))))
                    for q in range(1, self.moments):
                        features.append(np.power(np.abs(filtered_x), q))
        features = np.array(features)
        return features

    def _calculate_geoscattering(self, graph):
        """
        Calculating the features of a graph.

        Arg types:
            * **graph** *(NetworkX graph)* - A graph to be embedded.

        Return types:
            * **features** *(Numpy vector)* - The embedding of a single graph.
        """
        A_hat = self._get_normalized_adjacency(graph)
        Psi = self._calculate_wavelets(A_hat)
        X = self._create_node_feature_matrix(graph)
        zero_order_features = self._get_zero_order_features(X)
        first_order_features = self._get_first_order_features(Psi, X)
        second_order_features=self._get_second_order_features(Psi,X)
        features = np.concatenate([zero_order_features, first_order_features,second_order_features], axis=0)
        return features


class MyPreTransform(object):
    def __init__(self,config):
        self.model = GeoScattering_child()
        self.num_node = config.num_node
    def __call__(self, data):
        adj_mat = np.array(to_dense_adj(data.edge_index)[0])
        graph = networkx.convert_matrix.from_numpy_matrix(adj_mat)
        if not nx.is_connected(graph):
            graph=to_connected(graph)
        self.model.fit([graph])
        hks=self.model.get_embedding().reshape(eval_points,-1).transpose()
        #hks=torch.index_select(torch.from_numpy(hks), 1, indice)
        num = self.num_node - len(hks)
        m = torch.nn.ConstantPad2d((0, 0, 0, num), 0)
        mask = torch.cat((torch.ones((len(hks), eval_points)), torch.zeros((num, eval_points))), 0)
        hks = m(torch.from_numpy(hks)).float()[:, None, :]
        data.emb = hks
        data.mask = mask
        return data
