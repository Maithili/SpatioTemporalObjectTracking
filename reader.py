import json
import numpy as np
from math import ceil, floor
import torch
import random
from encoders import time_external
from torch.utils.data import DataLoader

from utils import visualize_routine

INTERACTIVE = False

def not_a_tree(original_edges, sparse_edges, nodes):
    num_parents = sparse_edges.sum(axis=-1)
    for i,num_p in enumerate(num_parents):
        if num_p>1:
            print(f'Node {nodes[i]} has parents : {list(np.array(nodes)[(np.argwhere(sparse_edges[i,:] > 0)).squeeze()])}')
            print(f'Node {nodes[i]} originally had parents : {list(np.array(nodes)[(np.argwhere(original_edges[i,:] > 0)).squeeze()])}')

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
            self.data, _ = self.make_pairwise(routines)
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
        additional_data = []
        self.time_min = torch.Tensor([float("Inf")])
        self.time_max = -torch.Tensor([float("Inf")])
        for routine in routines:
            nodes, edges, times, obj_in_use = routine
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
                    additional_data.append({'timestamp':time_external(prev_t), 'obj_in_use':obj_in_use[data_idx]})
                    # assert not(((edges_mask-edges[data_idx])<0).any())
                prev_edges = edges[data_idx]
                prev_nodes = nodes[data_idx]
                prev_t = t
        return pairwise_samples, additional_data
    

class RoutinesDataset():
    def __init__(self, data_path, 
                 classes_path, 
                 test_perc = 0.2, 
                 time_encoder = time_external, 
                 dt = 10,
                 batch_size = 1,
                 only_seen_edges = False):

        self.read_classes(classes_path)
        self.time_encoder = time_encoder
        
        self.params = {}
        self.params['dt'] = dt    # Overwrites the one present in classes
        self.params['batch_size'] = batch_size
        self.params['only_seen_edges'] = only_seen_edges

        # Read data
        if type(data_path) == tuple:
            self._train_data = self.read_data(data_path[0])
            self._test_data = self.read_data(data_path[1])
        else:
            self._all_data = self.read_data(data_path)
            random.shuffle(self._all_data)
            num_test = int(round(test_perc*len(self._all_data)))
            self._train_data = self._all_data[num_test:]
            self._test_data = self._all_data[:num_test]
            
        # Generate train and test loaders
        self.train = DataSplit(self._train_data, self.time_encoder, self.params['dt'], self.active_edges)
        self.test = DataSplit(self._test_data, self.time_encoder, self.params['dt'], self.active_edges)
        self.test_routines = DataSplit(self._test_data, self.time_encoder, self.params['dt'], self.active_edges, whole_routines=True)
        print(len(self._train_data),' routines and ',len(self.train),' examples in train split.')
        print(len(self._test_data),' routines and ',len(self.test),' examples in test split.')

        # Infer parameters for the model
        model_data = self.test.collate_fn([self.test[0]])
        self.params['n_nodes'] = model_data['edges'].size()[1]
        self.params['n_len'] = model_data['nodes'].size()[-1]
        self.params['c_len'] = model_data['context'].size()[-1]


    def read_data(self, data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)

        if INTERACTIVE:
            inp = input(f'Do you want to visualize all {len(data)} routines?')
        else:
            inp = 'n'
        viz = (inp == 'y')

        parsed_data = []
        for routine in data:
            if viz:
                visualize_routine(routine)
                inp = input(f'Do you want to visualize the next routine?')
                viz = (inp == 'y')
            nodes, edges = self.read_graphs(routine["graphs"])
            times = torch.Tensor(routine["times"])
            obj_in_use = routine["objects_in_use"]
            parsed_data.append((nodes, edges, times, obj_in_use))

        if self.params['only_seen_edges']:
            self.active_edges[self.seen_edges == 0] = 0
        self.active_edges = torch.Tensor(self.active_edges)
        
        return parsed_data

    def read_classes(self, classes_path):
        with open(classes_path, 'r') as f:
            classes = json.load(f)
        self.node_ids = [n['id'] for n in classes['nodes']]
        self.node_classes = [n['class_name'] for n in classes['nodes']]
        self.node_categories = [n['category'] for n in classes['nodes']]
        self.node_idx_from_id = {n['id']:i for i,n in enumerate(classes['nodes'])}

        # Diagonal nodes are always irrelevant
        self.active_edges = 1 - np.eye(len(classes['nodes']))
        # Rooms, furniture and appliances nodes don't move
        self.active_edges[np.where(np.array(self.node_categories) == "Rooms"),:] = 0
        self.active_edges[np.where(np.array(self.node_categories) == "Furniture"),:] = 0
        self.active_edges[np.where(np.array(self.node_categories) == "Decor"),:] = 0
        self.active_edges[np.where(np.array(self.node_categories) == "Appliances"),:] = 0
        self.seen_edges = np.zeros_like(self.active_edges)

        self.edge_keys = classes['edges']
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

    def get_train_loader(self):
        return DataLoader(self.train, num_workers=8, batch_size=self.params['batch_size'], sampler=self.train.sampler(), collate_fn=self.train.collate_fn)

    def get_test_loader(self):
        return DataLoader(self.test, num_workers=8, batch_size=self.params['batch_size'], sampler=self.test.sampler(), collate_fn=self.test.collate_fn)

    def get_single_example_test_loader(self):
        return DataLoader(self.test, num_workers=8, batch_size=1, sampler=self.test.sampler(), collate_fn=self.test.collate_fn)


def get_cooccurence_frequency(routines_dataset):
    all_edges = torch.concat([routine[1] for routine in routines_dataset._train_data], dim=0).sum(dim=0)
    all_edges += 1
    prior = all_edges/(all_edges.sum(dim=0))
    return prior


