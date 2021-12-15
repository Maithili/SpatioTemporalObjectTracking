import json
import numpy as np
from math import ceil, floor
import torch
import random
from encoders import time_sine_cosine
from torch.utils.data import DataLoader

from utils import visualize_routine

INTERACTIVE = False

def _densify(edges):
    dense_edges = edges.copy()
    for _ in range(edges.shape[-1]):
        new_edges = np.matmul(dense_edges, edges)
        new_edges = new_edges * (dense_edges==0)
        if (new_edges==0).all():
            break
        dense_edges += new_edges
    return dense_edges

def _sparsify(edges):
    dense_edges = _densify(edges.copy())
    remove = np.matmul(dense_edges, dense_edges)
    sparse_edges = dense_edges * (remove==0).astype(int)
    assert (sparse_edges.sum(axis=-1)).max() == 1, f"Matrix not really a tree \n{edges} \n{dense_edges} \n{sparse_edges}"
    return sparse_edges

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

class DataSplit():
    def __init__(self, routines, time_encoder, dt):
        self.time_encoder = time_encoder
        self.dt = dt
        self.data = self.make_pairwise(routines)
        self.collate_fn = CollateToDict(['edges', 'nodes', 'context', 'y_edges', 'y_nodes'])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx: int):
        return self.data[idx]
    def sampler(self):
        return None
    def make_pairwise(self, routines):
        pairwise_samples = []
        for routine in routines:
            nodes, edges, times = routine
            assert times[0]==min(times), 'Times need to be monotonically increasing. First element should be min.'
            assert times[-1]==max(times), 'Times need to be monotonically increasing. Last element should be max.'
            time_min = floor(times[0])
            time_max = ceil(times[-1])
            times = torch.cat([times,torch.Tensor([float("Inf")])], dim=-1)
            data_idx = -1
            prev_edges = None
            for t in range(time_min, time_max+1):
                if t >= times[data_idx+1]:
                    data_idx += 1
                if data_idx < 0:
                    continue
                if prev_edges is not None:
                    pairwise_samples.append((prev_edges, prev_nodes, self.time_encoder((t-1) * self.dt), edges[data_idx], nodes[data_idx]))
                prev_edges = edges[data_idx]
                prev_nodes = nodes[data_idx]
        random.shuffle(pairwise_samples)
        return pairwise_samples
    

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
                 only_dynamic_edges = False,
                 tree_formuation=False,
                 ignore_close_edges=True):

        self.data_path = data_path
        self.classes_path = classes_path
        self.time_encoder = time_encoder
        self.params = {}
        self.params['dt'] = dt    # Overwritten if present in classes
        self.params['edges_of_interest'] = edges_of_interest if edges_of_interest is not None else []
        self.params['sample_data'] = sample_data
        self.params['batch_size'] = batch_size
        self.params['avg_samples_per_routine'] = avg_samples_per_routine
        self.params['only_dynamic_edges'] = only_dynamic_edges
        self.params['tree_formuation'] = tree_formuation
        self.params['ignore_close_edges'] = ignore_close_edges

        # Read data
        self._alldata = self.read_data()

        # Split the data into train and test
        random.shuffle(self._alldata)
        num_test = int(round(test_perc*len(self._alldata)))
        self.test = DataSplit(self._alldata[:num_test], self.time_encoder, self.params['dt'])
        self.train = DataSplit(self._alldata[num_test:], self.time_encoder, self.params['dt'])
        print(len(self._alldata),' routines found in dataset.')
        print(len(self.train),' examples in train split.')
        print(len(self.test),' examples in test split.')

        # Infer parameters for the model
        model_data = self.test.collate_fn([self.test[0]])
        self.params['n_nodes'] = model_data['edges'].size()[1]
        self.params['n_len'] = model_data['nodes'].size()[-1]
        self.params['e_len'] = 1
        self.params['c_len'] = model_data['context'].size()[-1]

    def read_data(self):
        with open(self.classes_path, 'r') as f:
            classes = json.load(f)
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        self.read_classes(classes)

        if INTERACTIVE:
            inp = input(f'Do you want to visualize all {len(data)} routines?')
        else:
            inp = 'n'
        viz = (inp == 'y')

        training_data = []
        for routine in data:
            if viz:
                visualize_routine(routine, dt=self.params['dt'])
                inp = input(f'Do you want to visualize the next routine?')
                viz = (inp == 'y')
            nodes, edges = self.read_graphs(routine["graphs"])
            times = torch.Tensor(routine["times"])
            training_data.append((nodes, edges, times))

        return training_data

    def read_classes(self, classes):
        self.node_keys = [n['id'] for n in classes['nodes']]
        self.node_classes = [n['class_name'] for n in classes['nodes']]
        self.node_categories = [n['category'] for n in classes['nodes']]
        self.node_states = {}
        for i,state_pairs in enumerate(classes['node_states']):
            self.node_states[state_pairs[0]] = np.zeros(len(classes['node_states']))
            self.node_states[state_pairs[0]][i] = -1
            self.node_states[state_pairs[1]] = np.zeros(len(classes['node_states']))
            self.node_states[state_pairs[1]][i] = 1
        self.edge_keys = classes['edges']
        if self.params['ignore_close_edges']:
            self.edge_keys.remove("CLOSE")
        if 'dt' in classes:
            self.params['dt'] = classes['dt']
        static = lambda category : category in ["Furniture", "Room"]
        self.static_nodes = [n['id'] for n in classes['nodes'] if static(n['category'])]

    def read_graphs(self, graphs):
        nodes = graphs[0]['nodes']
        node_ids = [n['id'] for n in nodes]

        self.params['n_class_len'] = len(self.node_keys)
        self.params['n_state_len'] = int(round(len(self.node_states)/2))
        node_feature_len = self.params['n_class_len'] + self.params['n_state_len']
        node_features = np.zeros((len(graphs), len(nodes), node_feature_len))
        edge_features = np.zeros((len(graphs), len(node_ids), len(node_ids), 1))
        for i,graph in enumerate(graphs):
            graph_nodes = [[node for node in graph['nodes'] if node['id'] == nid][0] for nid in node_ids]
            for j,n1 in enumerate(node_ids):
                node_features[i,j,:] = self.encode_node(graph_nodes[j])
                for k,n2 in enumerate(node_ids):
                    if self.params['only_dynamic_edges'] and (n1 in self.static_nodes) and (n2 in self.static_nodes):
                        continue
                    edge_features[i,j,k,:]= self.encode_edge(self.get_edges(graph, n1, n2))
            if self.params['tree_formuation']:
                edge_features[i,:,:,0] = _sparsify(edge_features[i,:,:,0])
        return torch.Tensor(node_features), torch.Tensor(edge_features)

    def get_edges(self, graph, n_id1, n_id2):
        edges = [e for e in graph['edges'] if e['from_id']==n_id1 and e['to_id']==n_id2]
        return edges

    def encode_edge(self, edges):
        valid = lambda rel: rel in [e['relation_type'] for e in edges]
        encoding = 0
        for c in self.edge_keys:
            if valid(c):
                encoding = 1
                break
        return encoding

    def encode_node(self, node):
        node_class = np.array(self.node_keys) == node['id']
        node_state = sum([self.node_states[s] for s in node['states']])
        return np.concatenate((node_class,node_state))

    def get_train_loader(self):
        return DataLoader(self.train, num_workers=8, batch_size=self.params['batch_size'], sampler=self.train.sampler(), collate_fn=self.train.collate_fn)

    def get_test_loader(self):
        return DataLoader(self.test, num_workers=8, batch_size=len(self.test), sampler=self.test.sampler(), collate_fn=self.test.collate_fn)

    def get_single_example_test_loader(self):
        return DataLoader(self.test, num_workers=8, batch_size=1, sampler=self.test.sampler(), collate_fn=self.test.collate_fn)

    def get_edges_of_interest(self):
        edges = {}
        for from_node, relation, to_node in self.params['edges_of_interest']:
            from_feat = self.node_classes.index(from_node)
            to_feat = self.node_classes.index(to_node)
            rel_idx = self.edge_keys.index(relation)
            name = ' '.join([from_node, relation, to_node])
            edges[name] = (from_feat,to_feat,rel_idx)
        return edges
