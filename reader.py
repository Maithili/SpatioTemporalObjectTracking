import json
import numpy as np
from os import path as osp
from math import ceil, floor
from numpy.core.fromnumeric import argmax
import torch
import random
from encoders import time_sine_cosine
from torch.utils.data import WeightedRandomSampler, DataLoader

class CollateToDict():
    def __init__(self, dict_labels):
        self.dict_labels = dict_labels

    def __call__(self, tensor_tuple):
        data = {label:torch.Tensor() for label in self.dict_labels}
        for tensors in tensor_tuple:
            assert len(tensors) == len(self.dict_labels), "Wrong number of labels provided to the collate function!"
            for label, tensor in zip(self.dict_labels, tensors):
                data[label]=torch.cat([data[label], tensor.unsqueeze(0)], dim=0)
        return data


class RoutinesCollateFn():
    def __init__(self, time_encoder, sampling=True):
        self.time_encoder = time_encoder
        if sampling:
            self.get_datapoint = self.get_datapoint_interpolate
        else:
            self.get_datapoint = self.get_datapoint_choose

    def get_datapoint_interpolate(self, nodes, edges, times):
        i = random.random()*len(times)
        j = random.random()*len(times)
        times = torch.cat([times, torch.Tensor([times[-1]+ (times[-1] - times[-2])])])
        if i > j:
            t=i
            i=j
            j=t
        def interp_time(sample):
            idx = floor(sample)
            f = sample - idx
            t = f * times[idx+1] + (1-f) * times[idx]
            return idx, t
        i,time_i = interp_time(i) * self.params['dt']
        j,time_j = interp_time(j) * self.params['dt']
        return  (edges[i], nodes, self.time_encoder(time_i), self.time_encoder(time_j), edges[j])

    def get_datapoint_choose(self, nodes, edges, times):
        i = random.randrange(0, len(times))
        j = random.randrange(0, len(times))
        while (i>=j):
            i = random.randrange(0, len(times))
            j = random.randrange(0, len(times))
        return  (edges[i], nodes, self.time_encoder(times[i]), self.time_encoder(times[j]), edges[j])

    def __call__(self, routine_list):
        data = {}

        data['edges'] = torch.Tensor()
        data['nodes'] = torch.Tensor()
        data['context'] = torch.Tensor()
        data['y'] = torch.Tensor()

        for routine in routine_list:
            e1, n, c1, c2, e2 = self.get_datapoint(routine[0], routine[1], routine[2])
            data['edges'] = torch.cat([data['edges'], e1.unsqueeze(0)], dim=0)
            data['nodes'] = torch.cat([data['nodes'], n.unsqueeze(0)], dim=0)
            c = torch.cat([c1,c2], dim=-1)
            data['context'] = torch.cat([data['context'], c.unsqueeze(0)], dim=0)
            data['y'] = torch.cat([data['y'], e2.unsqueeze(0)], dim=0)

        return data

class DataSplit():
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx: int):
        return self.data[idx]
    def weights(self):
        graphs_in_routine = torch.Tensor([len(d[1]) for d in self.data])
        return graphs_in_routine/graphs_in_routine.sum()


class RoutinesDataset():
    def __init__(self, data_path: str = 'data/example/sample.json', 
                 classes_path: str = 'data/example/classes.json', 
                 test_perc = 0.2, 
                 time_encoder = time_sine_cosine, 
                 dt = 10,
                 edges_of_interest = None,
                 sample_data = True,
                 batch_size = 1,
                 avg_samples_per_routine = 1,
                 sequential_prediction = True,
                 time_range = ((6,0),(24,0))):

        self.data_path = data_path
        self.classes_path = classes_path
        self.time_encoder = time_encoder
        self.params = {}
        self.params['dt'] = dt
        self.params['edges_of_interest'] = edges_of_interest if edges_of_interest is not None else []
        self.params['sample_data'] = sample_data
        self.params['batch_size'] = batch_size
        self.params['avg_samples_per_routine'] = avg_samples_per_routine
        self.params['time_range'] = time_range
        def noCollate():
            raise Exception('No collate function defined')
        self.collate_fn = noCollate

        ## Read and divide data
        self._alldata = self.read_data()
        print(len(self._alldata),' examples found in dataset.')
        # Infer parameters from data
        nodes, edges, times = self._alldata[0]
        # Split the data into train and test
        random.shuffle(self._alldata)
        num_test = int(round(test_perc*len(self._alldata)))
        if sequential_prediction:
            self.test = DataSplit(self.make_pairwise(self._alldata[:num_test]))
            self.train = DataSplit(self.make_pairwise(self._alldata[num_test:]))
        else:
            self.test = DataSplit(self._alldata[:num_test])
            self.train = DataSplit(self._alldata[num_test:])
            self.collate_fn = RoutinesCollateFn(self.time_encoder, sampling=self.params['sample_data'])
        print(len(self.train),' examples in train split.')
        print(len(self.test),' examples in test split.')
        model_data = self.collate_fn([self.train[0]])
        self.params['n_nodes'] = model_data['edges'].size()[1]
        self.params['n_len'] = model_data['nodes'].size()[-1]
        self.params['e_len'] = model_data['edges'].size()[-1]
        self.params['c_len'] = model_data['context'].size()[-1]
        
    def log_random_loss(self, analyzers):
        for d in self.train:
            random_out = torch.nn.Sigmoid()(torch.randn(self.params['n_nodes'], self.params['n_nodes'], self.params['e_len']))
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
            nodes, edges = self.read_graphs(routine["graphs"])
            times = torch.Tensor(routine["times"])
            training_data.append((nodes, edges, times))

        return training_data

    def read_classes(self, classes):
        self.node_keys = [n['id'] for n in classes['nodes']]
        self.node_classes = [n['class_name'] for n in classes['nodes']]
        self.edge_keys = classes['edges']
        if 'dt' in classes:
            self.params['dt'] = classes['dt']
        static = lambda category : category in ["Furniture", "Room"]
        self.static_nodes = [n['id'] for n in classes['nodes'] if static(n['category'])]

    def read_graphs(self, graphs):
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

        return torch.Tensor(node_features), torch.Tensor(edge_features)

    def make_pairwise(self, data):
        pairwise_samples = []
        time_min = int((self.params['time_range'][0][1] + self.params['time_range'][0][0]*60)/self.params['dt'])
        time_max = int((self.params['time_range'][1][1] + self.params['time_range'][1][0]*60)/self.params['dt'])
        for routine in data:
            nodes, edges, times = routine
            times = times % (24*60/self.params['dt'])
            time_min = floor(min(times))
            time_max = ceil(max(times))
            times = torch.cat([times,torch.Tensor([time_max+1])], dim=-1)
            data_idx = -1
            prev_datapoint = None
            for t in range(time_min, time_max):
                if t >= times[data_idx+1]:
                    data_idx += 1
                if data_idx<0:
                    continue
                if prev_datapoint:
                    pairwise_samples.append((prev_datapoint, nodes, self.time_encoder((t-1) * self.params['dt']), edges[data_idx]))
                prev_datapoint = edges[data_idx]
        self.collate_fn = CollateToDict(['edges', 'nodes', 'context', 'y'])
        return pairwise_samples


    def get_edges(self, graph, n_id1, n_id2):
        edges = [e for e in graph['edges'] if e['from_id']==n_id1 and e['to_id']==n_id2]
        return edges

    def encode_edge(self, edges):
        valid = lambda rel: rel in [e['relation_type'] for e in edges]
        encoding = [1 if valid(c) else 0 for c in self.edge_keys]
        return encoding

    def encode_node(self, node):
        return np.array(self.node_keys) == node['id']

    def get_sampler(self, weights):
        return WeightedRandomSampler(weights=weights,
                                    num_samples=weights.size()[-1]*self.params['avg_samples_per_routine'],
                                    replacement=True
                                    )

    def get_train_loader(self):
        return DataLoader(self.train, num_workers=8, batch_size=self.params['batch_size'], sampler=self.get_sampler(self.train.weights()), collate_fn=self.collate_fn)

    def get_test_loader(self):
        return DataLoader(self.test, num_workers=8, batch_size=1, sampler=self.get_sampler(self.test.weights()), collate_fn=self.collate_fn)

    def get_edges_of_interest(self):
        edges = {}
        for from_node, relation, to_node in self.params['edges_of_interest']:
            from_feat = self.node_classes.index(from_node)
            to_feat = self.node_classes.index(to_node)
            rel_idx = self.edge_keys.index(relation)
            name = ' '.join([from_node, relation, to_node])
            edges[name] = (from_feat,to_feat,rel_idx)
        return edges
