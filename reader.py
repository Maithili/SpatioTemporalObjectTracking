import json
import numpy as np
from os import path as osp
import torch
import random
from encoders import time_sine_cosine


def pairwise_data(nodes, edges, contexts):
    n = len(edges)

    data = []
    for i in range(n):
        for j in range(i+1,n):
            data.append((edges[i], nodes, contexts[i], contexts[j], edges[j]))
    return data


class DataSplit():
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx: int):
        return self.data[idx]

class RoutinesDataset():
    def __init__(self, data_path: str = 'data/example/sample.json', classes_path: str = 'data/example/classes.json', test_perc = 0.2, time_encoder=time_sine_cosine):
        self.data_path = data_path
        self.classes_path = classes_path
        self.time_encoder = time_encoder
        self._alldata = self.read_data()
        print(len(self._alldata),' examples found in dataset.')
        # Infer parameters from data
        edges, nodes, context_curr, context_query, y = self._alldata[0]
        self.n_nodes = edges.size()[1]
        self.n_len = nodes.size()[1]
        self.e_len = edges.size()[-1]
        self.c_len = context_curr.size()[0]
        # Split the data into train and test
        random.shuffle(self._alldata)
        num_test = int(round(test_perc*len(self._alldata)))
        self.test = DataSplit(self._alldata[:num_test])
        self.train = DataSplit(self._alldata[num_test:])
        print(len(self.train),' examples in train split.')
        print(len(self.test),' examples in test split.')
        
    def log_random_loss(self, analyzers):
        for d in self.train:
            random_out = torch.nn.Sigmoid()(torch.randn(self.n_nodes, self.n_nodes, self.e_len))
            y = d[4]
            losses = torch.nn.BCELoss()(random_out, y)
            for analyzer in analyzers:
                self.log('Random: '+analyzer.name(), analyzer(losses, x_edges=random_out, y_edges=y))

    def read_data(self):
        with open(self.classes_path, 'r') as f:
            classes = json.load(f)
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        self.read_classes(classes)

        training_data = []
        for routine in data:
            nodes, edges, contexts = self.read_routine(routine["graphs"], routine["times"])
            training_data += pairwise_data(nodes, edges, contexts)

        return training_data

    def read_classes(self, classes):
        self.node_keys = [n['id'] for n in classes['nodes']]
        self.edge_keys = classes['edges']
        static = lambda category : category in ["Furniture", "Room"]
        self.static_nodes = [n['id'] for n in classes['nodes'] if static(n['category'])]

    def read_routine(self, graphs, times):
        nodes = graphs[0]['nodes']
        node_ids = [n['id'] for n in nodes]

        node_features = [None] * len(nodes)
        for i,nid in enumerate(nodes):
            node_features[i] = self.encode_node(nid)
        node_features = np.array(node_features)

        edge_features = np.zeros((len(graphs), len(node_ids), len(node_ids), len(self.edge_keys)))
        for i,graph in enumerate(graphs):
            for j,n1 in enumerate(node_ids):
                for k,n2 in enumerate(node_ids):
                    edge_features[i,j,k,:]= self.encode_edge(self.get_edges(graph, n1, n2))

        context = [self.time_encoder(t*10) for t in times]

        return torch.Tensor(node_features), torch.Tensor(edge_features), torch.Tensor(context)

    def get_edges(self, graph, n_id1, n_id2):
        edges = [e for e in graph['edges'] if e['from_id']==n_id1 and e['to_id']==n_id2]
        return edges

    def encode_edge(self, edges):
        valid = lambda rel: rel in [e['relation_type'] for e in edges]
        encoding = [1 if valid(c) else 0 for c in self.edge_keys]
        return encoding

    def encode_node(self, node):
        return np.array(self.node_keys) == node['id']

    def get_train_loader(self):
        return torch.utils.data.DataLoader(self.train, num_workers=8, batch_size=10)

    def get_test_loader(self):
        return torch.utils.data.DataLoader(self.test, num_workers=8, batch_size=10)

    def get_node_classes(self):
        return self.node_keys

    def get_edge_classes(self):
        return self.edge_keys

    def get_static_nodes(self):
        return self.static_nodes
