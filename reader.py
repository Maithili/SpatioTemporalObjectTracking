import json
import numpy as np
from math import ceil, floor
from numpy.core.numeric import zeros_like
import torch
import random
from encoders import time_sine_cosine
from torch.utils.data import DataLoader

from utils import visualize_routine

INTERACTIVE = False

def not_a_tree(original_edges, sparse_edges, nodes):
    num_parents = sparse_edges.sum(axis=-1)
    for i,num_p in enumerate(num_parents):
        if num_p>1:
            print(f'Node {nodes[i]} has parents : {nodes[(np.argwhere(sparse_edges[i,:] > 0)).squeeze()]}')
            print(f'Node {nodes[i]} originally had parents : {nodes[(np.argwhere(original_edges[i,:] > 0)).squeeze()]}')

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
    def __init__(self, routines, time_encoder, dt, active_edges, whole_routines=False):
        self.time_encoder = time_encoder
        self.dt = dt
        self.active_edges = active_edges
        if whole_routines:
            self.data = [self.make_pairwise([routine]) for routine in routines]
        else:
            self.data = self.make_pairwise(routines)
            random.shuffle(self.data)
        self.collate_fn = CollateToDict(['edges', 'nodes', 'context', 'y_edges', 'y_nodes', 'dynamic_edges_mask'])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx: int):
        return self.data[idx]
    def sampler(self):
        return None
    def make_pairwise(self, routines):
        pairwise_samples = []
        self.time_min = torch.Tensor([float("Inf")])
        self.time_max = -torch.Tensor([float("Inf")])
        for routine in routines:
            nodes, edges, times = routine
            assert times[0]==min(times), 'Times need to be monotonically increasing. First element should be min.'
            assert times[-1]==max(times), 'Times need to be monotonically increasing. Last element should be max.'
            time_min = floor(times[0]/self.dt)*self.dt
            time_max = ceil(times[-1]) + self.dt
            if time_min < self.time_min: self.time_min = time_min
            if time_max > self.time_max: self.time_max = time_max
            times = torch.cat([times,torch.Tensor([float("Inf")])], dim=-1)
            data_idx = -1
            prev_edges = None
            for t in range(time_min, time_max, self.dt):
                while t >= times[data_idx+1]:
                    data_idx += 1
                if data_idx < 0:
                    continue
                if prev_edges is not None:
                    edges_mask = self.active_edges
                    pairwise_samples.append((prev_edges, prev_nodes, self.time_encoder(prev_t), edges[data_idx], nodes[data_idx], edges_mask))
                    # assert not(((edges_mask-edges[data_idx])<0).any())
                prev_edges = edges[data_idx]
                prev_nodes = nodes[data_idx]
                prev_t = t
        return pairwise_samples
    

class RoutinesDataset():
    def __init__(self, data_path: str = 'data/example/sample.json', 
                 classes_path: str = 'data/example/classes.json', 
                 test_perc = 0.2, 
                 time_encoder = time_sine_cosine, 
                 dt = 10,
                 edges_of_interest = None,
                 batch_size = 1,
                 only_seen_edges = False):

        self.data_path = data_path
        self.classes_path = classes_path
        self.time_encoder = time_encoder
        self.params = {}
        self.params['dt'] = dt    # Overwritten if present in classes
        self.params['edges_of_interest'] = edges_of_interest if edges_of_interest is not None else []
        self.params['batch_size'] = batch_size
        self.params['only_seen_edges'] = only_seen_edges

        # Read data
        self._alldata = self.read_data()

        # Split the data into train and test
        random.shuffle(self._alldata)
        num_test = int(round(test_perc*len(self._alldata)))
        self.test = DataSplit(self._alldata[:num_test], self.time_encoder, self.params['dt'], self.active_edges)
        self.train = DataSplit(self._alldata[num_test:], self.time_encoder, self.params['dt'], self.active_edges)
        self.all_routines = DataSplit(self._alldata, self.time_encoder, self.params['dt'], self.active_edges, whole_routines=True)
        print(len(self._alldata),' routines found in dataset.')
        print(len(self.train),' examples in train split.')
        print(len(self.test),' examples in test split.')

        # Infer parameters for the model
        model_data = self.test.collate_fn([self.test[0]])
        self.params['n_nodes'] = model_data['edges'].size()[1]
        self.params['n_len'] = model_data['nodes'].size()[-1]
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

        if self.params['only_seen_edges']:
            self.active_edges[self.seen_edges == 0] = 0
        self.active_edges = torch.Tensor(self.active_edges)
        
        return training_data

    def read_classes(self, classes):
        self.node_ids = [n['id'] for n in classes['nodes']]
        self.node_classes = [n['class_name'] for n in classes['nodes']]
        self.node_categories = [n['category'] for n in classes['nodes']]
        self.node_idx_from_id = {n['id']:i for i,n in enumerate(classes['nodes'])}
        # Diagonal nodes are always irrelevant
        self.active_edges = 1 - np.eye(len(classes['nodes']))
        # Rooms, furniture and appliances nodes don't move
        self.active_edges[np.where(np.array(self.node_categories) == "Rooms"),:] = 0
        self.active_edges[np.where(np.array(self.node_categories) == "Furniture"),:] = 0
        self.active_edges[np.where(np.array(self.node_categories) == "Appliances"),:] = 0
        self.seen_edges = np.zeros_like(self.active_edges)

        self.edge_keys = classes['edges']
        self.edge_keys.remove("CLOSE")
        if 'dt' in classes:
            self.params['dt'] = classes['dt']
        static = lambda category : category in ["Furniture", "Room"]
        self.static_nodes = [n['id'] for n in classes['nodes'] if static(n['category'])]

    def read_graphs(self, graphs):
        node_features = np.zeros((len(graphs), len(self.node_ids), len(self.node_ids)))
        edge_features = np.zeros((len(graphs), len(self.node_ids), len(self.node_ids)))
        for i,graph in enumerate(graphs):
            node_features[i,:,:len(self.node_ids)] = np.eye(len(self.node_ids))
            for e in graph['edges']:
                if e['relation_type'] in self.edge_keys:
                    edge_features[i,self.node_idx_from_id[e['from_id']],self.node_idx_from_id[e['to_id']]] = 1
            original_edges = edge_features[i,:,:]
            edge_features[i,:,:] = _sparsify(edge_features[i,:,:])
            if (edge_features[i,:,:].sum(axis=-1)).max() != 1:
                print(f"Matrix {i} not really a tree \n{edge_features[i,:,:]}")
                not_a_tree(original_edges, edge_features[i,:,:], self.node_classes)
            assert (edge_features[i,:,:].sum(axis=-1)).max() == 1, f"Matrix {i} not really a tree \n{edge_features[i,:,:]}"
            if self.params['only_seen_edges']:
                self.seen_edges[:,:] += edge_features[i,:,:]
        return torch.Tensor(node_features), torch.Tensor(edge_features)

    def get_edges(self, graph, n_id1, n_id2):
        edges = [e for e in graph['edges'] if e['from_id']==n_id1 and e['to_id']==n_id2]
        return edges

    def encode_edge(self, edges):
        valid = any([e['relation_type'] in self.edge_keys for e in edges])
        encoding = 1 if valid else 0
        return encoding

    def encode_node(self, node):
        node_class = np.array(self.node_ids) == node['id']
        return node_class

    def get_class_from_id(self, id):
        return self.node_ids.index(id)

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
