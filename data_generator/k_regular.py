import numpy as np
import torch
import random
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.utils.sparse as sp
import networkx as nx
np.random.seed(0)
random.seed(0)
ns = [8, 10, 12, 12, 14, 14, 16, 16, 18, 18]
rs = [4, 4,  4,   5, 4, 6,   4,  6,  5,  8]

def count_clique(adj_matrix, n):
    count = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for t in range(n):
                    if i < j < k < t and adj_matrix[i][j] == 1 and adj_matrix[i][k] == 1 and adj_matrix[j][
                        k] == 1 and adj_matrix[i][t] == 1 and adj_matrix[j][t] == 1 and adj_matrix[k][t] == 1:
                        count += 1
    return count * 24.0 / (n * (n - 1) * (n - 2) * (n - 3))



def generate_graph(n, r, seed):
    rand_graph = nx.random_regular_graph(r, n, seed=seed)
    A = nx.to_numpy_matrix(rand_graph)
    num_triangles = sum(nx.triangles(rand_graph).values()) / 3
    return A, num_triangles


class KRegular(InMemoryDataset):
    def __init__(self, root,which,transform=None, pre_transform=None,
                 pre_filter=None):
        self.which = which
        super(KRegular, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'k_cycle_raw.pt'

    @property
    def processed_file_names(self):
        return 'k_cycle.pt'
    
    def generate(self):
        adj_matrix = []
        labels = []
        n = ns[self.which]
        r = rs[self.which]
        for i in range(500):
            g_matrix, num_triangles = generate_graph(n=n, r=r, seed=i)
            g_matrix = torch.tensor(g_matrix).float()
            adj_matrix.append(g_matrix)
            count = []
            count.append(num_triangles)
            count.append(count_clique(g_matrix, n))
            labels.append(count)

        c = list(zip(adj_matrix, labels))
        random.shuffle(c)
        self.adj_matrix, self.labels = zip(*c)
        self.labels = torch.tensor(self.labels)
        self.labels = (self.labels - self.labels.mean()) / self.labels.std()

    def download(self):
        self.generate()
        data_list = []
        for i in range(500):
            data = Data()
            data.y = self.labels[i].view(-1, 2)
            edge_index = sp.dense_to_sparse(self.adj_matrix[i])[0]
            data.edge_index = edge_index
            data.name = i
            data.num_nodes = len(self.adj_matrix[i])
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.raw_paths[0])
    

    def process(self):
        self.data, self.slices = torch.load(self.raw_paths[0])
        data_list = [self.get(i) for i in range(len(self))]

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return