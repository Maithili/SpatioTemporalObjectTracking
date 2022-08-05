import json
import os
import shutil
import argparse
import numpy as np
from math import ceil, floor
import torch
import random
from encoders import time_external
from torch.utils.data import DataLoader

from graph_visualizations import visualize_routine, visualize_parsed_routine, visualize_raw_and_parsed_routine

INTERACTIVE = False

ROLES = {
    'cleanliness' : ["CLEAN", "DIRTY"],
    'cookedness' : ["UNCOOKED", "COOKED"],
    'fullness' : ["EMPTY", "FULL"],
    'openness' : ["CLOSED", "OPEN"],
    'pluggedin' : ["PLUGGED_IN", "PLUGGED_OUT"],
    'turnedon' : ["OFF", "ON"],
    'wetness' : ["DRY", "WET"]
    }


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
            assert len(tensors) == len(self.dict_labels), f"Wrong number of labels provided to the collate function! {len(self.dict_labels)} labels for {len(tensors)} tensors"
            for label, tensor in zip(self.dict_labels, tensors):
                data[label]=torch.cat([data[label], tensor.unsqueeze(0)], dim=0)
        return data

class DataSplit():
    def __init__(self, routines_dir, time_encoder, dt, active_edges, idx_map, whole_routines=False, max_num_files=None):
        self.time_encoder = time_encoder
        self.dt = dt
        self.active_edges = active_edges
        self.idx_map = idx_map
        self.routines_dir = routines_dir
        self.whole_routines = whole_routines
        self.collate_fn = CollateToDict(['edges', 'nodes', 'context', 'y_edges', 'y_nodes', 'dynamic_edges_mask', 'time', 'change_type'])
        self.files = [name for name in os.listdir(self.routines_dir) if os.path.isfile(os.path.join(self.routines_dir, name))]
        self.files.sort()
        if max_num_files is not None:
            assert max_num_files <= len(self.files)
            self.files = self.files[:max_num_files]
            self.idx_map = [(f,i) for f,i in self.idx_map if f+'.pt' in self.files]

    def num_samples(self):
        return len(self.idx_map)
    def num_routines(self):
        return len(self.files)
    def __len__(self):
        return self.num_routines() if self.whole_routines else self.num_samples()

    def get_sample(self, idx: int):
        filename, sample_idx = self.idx_map[idx]
        data_list = torch.load(os.path.join(self.routines_dir, filename+'.pt')) 
        sample = data_list[sample_idx]
        return torch.Tensor(sample['prev_edges']), sample['prev_nodes'], self.time_encoder(sample['time']), torch.Tensor(sample['edges']), sample['nodes'], self.active_edges, torch.tensor(sample['time']).to(int), sample['change_type']
    def get_routine(self, idx: int):
        data_list = torch.load(os.path.join(self.routines_dir, self.files[idx])) 
        samples = [(torch.Tensor(sample['prev_edges']), sample['prev_nodes'], self.time_encoder(sample['time']), torch.Tensor(sample['edges']), sample['nodes'], self.active_edges, torch.tensor(sample['time']).to(int), sample['change_type']) for sample in data_list]
        additional_info = {'timestamp':[time_external(sample['time']) for sample in data_list], 'active_nodes':self.active_edges.sum(-1) > 0}
        return samples, additional_info
    def __getitem__(self, idx: int):
        return self.get_routine(idx) if self.whole_routines else self.get_sample(idx)

    def get_edges_and_time(self, idx:int):
        filename, sample_idx = self.idx_map[idx]
        data_list = torch.load(os.path.join(self.routines_dir, filename+'.pt')) 
        sample = data_list[sample_idx]
        return sample['edges'], sample['time']
    
    def sampler(self):
        return None




class RoutinesDataset():
    def __init__(self, data_path, 
                 time_encoder = time_external, 
                 batch_size = 1,
                 only_seen_edges = False,
                 max_routines = (None, None)):

        with open(os.path.join(data_path, 'common_data.json')) as f:
            self.common_data = json.load(f)

        self.time_encoder = time_encoder
        
        self.params = {}
        self.params['dt'] = self.common_data['info']['dt']
        self.params['batch_size'] = batch_size

        if only_seen_edges:
            self.active_edges = torch.load(os.path.join(data_path, 'seen_edges.pt')) #, map_location=lambda storage, loc: storage.cuda(1))
        else:
            self.active_edges = torch.load(os.path.join(data_path, 'nonstatic_edges.pt')) #, map_location=lambda storage, loc: storage.cuda(1))
        
        self.home_graph = torch.load(os.path.join(data_path, 'home_graph.pt'))

        # self.node_ids = self.common_data['node_ids']
        # self.node_idx_from_id = {int(k):v for k,v in self.common_data['node_idx_from_id'].items()}
        # self.edge_keys = self.common_data['edge_keys']
        # self.static_nodes = self.common_data['static_nodes']
        
            
        # Generate train and test loaders
        self.train = DataSplit(os.path.join(data_path,'train'), self.time_encoder, self.params['dt'], self.active_edges, self.common_data['train_data_index_list'], max_num_files=max_routines[0])
        self.test = DataSplit(os.path.join(data_path,'test'), self.time_encoder, self.params['dt'], self.active_edges, self.common_data['test_data_index_list'], max_num_files=max_routines[1])
        self.test_routines = DataSplit(os.path.join(data_path,'test'), self.time_encoder, self.params['dt'], self.active_edges, self.common_data['test_data_index_list'], whole_routines=True)
        print(len(self.train),' examples in train split.')
        print(len(self.test),' examples in test split.')

        # Infer parameters for the model
        model_data = self.test.collate_fn([self.test[0]])
        self.params['n_nodes'] = model_data['edges'].size()[1]
        self.params['c_len'] = model_data['context'].size()[-1]
        self.params['num_classes'] = len(self.common_data['node_classes'])
        self.params['num_categories'] = len(self.common_data['node_categories'])
        self.params['num_states'] = len(self.common_data['node_states'])


    def get_train_loader(self):
        return DataLoader(self.train, num_workers=8, batch_size=self.params['batch_size'], sampler=self.train.sampler(), collate_fn=self.train.collate_fn)

    def get_test_loader(self):
        return DataLoader(self.test, num_workers=8, batch_size=self.params['batch_size'], sampler=self.test.sampler(), collate_fn=self.test.collate_fn)

    def get_single_example_test_loader(self):
        return DataLoader(self.test, num_workers=8, batch_size=1, sampler=self.test.sampler(), collate_fn=self.test.collate_fn)
    

def get_cooccurence_frequency(routines_dataset):
    # one-smoothing
    all_edges = torch.ones_like(routines_dataset.train[0][0])
    for r in routines_dataset.train:
        all_edges = all_edges + r[0]
    prior = all_edges/(all_edges.sum(dim=0))/(len(routines_dataset.train) + 1)
    print('Size of coccurence prior : ',prior.size())
    return prior

def get_spectral_components(routines_dataset, periods_mins):
    reals = [torch.zeros_like(routines_dataset.train[0][0]) for _ in periods_mins]
    imags = [torch.zeros_like(routines_dataset.train[0][0]) for _ in periods_mins]
    for idx in range(len(routines_dataset.train)):
        edges, time = routines_dataset.train.get_edges_and_time(idx)
        for harmonic, period in enumerate(periods_mins):
            reals[harmonic] += edges * np.cos(2*np.pi*time/period)
            imags[harmonic] += edges * np.sin(2*np.pi*time/period)
    components = []
    for r,i,p in zip(reals, imags, periods_mins):
        components.append({'amplitude': (np.sqrt(np.square(r)+np.square(i))/len(routines_dataset.train)), 'phase': (np.arctan2(i,r)), 'period': p})
    print('Size of FreMeN prior : ',str([(comp['amplitude'].size(),comp['phase'].size()) for comp in components]))
    return components



class ProcessDataset():
    def __init__(self, data_path, 
                 classes_path,
                 info_path, 
                 output_path):

        output_train_path = os.path.join(output_path, 'train')
        output_test_path = os.path.join(output_path, 'test')
        if os.path.exists(output_train_path): 
            overwrite = input('Dataset seems to already exist!! Overwrite? (y/n)')
            if overwrite != 'y': raise InterruptedError('Cancelling data processing...')
            shutil.rmtree(output_path)
        os.makedirs(output_train_path)
        if os.path.exists(output_test_path): 
            overwrite = input('Dataset seems to already exist!! Overwrite? (y/n)')
            if overwrite != 'y': raise InterruptedError('Cancelling data processing...')
            shutil.rmtree(output_path)
        os.makedirs(output_test_path)

        self.common_data = {}
        with open(info_path) as f:
            self.common_data['info'] = json.load(f)
        self.dt = self.common_data['info']['dt']
        self.read_classes(classes_path)


        assert type(data_path) == tuple
        self.common_data['train_data_index_list'] = self.read_data(data_path[0], output_train_path)
        self.common_data['test_data_index_list'] = self.read_data(data_path[1], output_test_path)
        json.dump(self.common_data, open(os.path.join(output_path, 'common_data.json'), 'w'), indent=4)
        torch.save(self.seen_edges, os.path.join(output_path, 'seen_edges.pt'))
        torch.save(self.nonstatic_edges, os.path.join(output_path, 'nonstatic_edges.pt'))
        torch.save(self.home_graph, os.path.join(output_path, 'home_graph.pt'))

    def feature_from_node(self, node=None):
        if node == None:
            empty_feature = self.state_features["NONE"]
            empty_feature = np.concatenate([np.array([0, 0]),empty_feature], axis=-1).reshape(1,-1)
            return empty_feature

        if len(node['states']) > 0:
            feature = sum(self.state_features[state] for state in node['states'])
        else:
            feature = self.state_features["NONE"]
        cls = self.common_data['node_roles']['class'].index(node['class_name'])
        cat = self.common_data['node_roles']['category'].index(node['category'])
        feature = np.concatenate([np.array([cls, cat]),feature], axis=-1)
        return feature.reshape(1,-1)

    def decode_states(self, state_array):
        states = set()
        for idx_onebased, (role, values) in zip(state_array, ROLES.items()):
            if idx_onebased > 0:
                states.add(values[idx_onebased-1])
        return states

    def read_classes(self, classes_path):
        with open(classes_path, 'r') as f:
            classes = json.load(f)
        self.common_data['node_classes'] = [n['class_name'] for n in classes['nodes']]
        self.common_data['node_categories'] = [n['category'] for n in classes['nodes']]
        self.common_data['node_states'] = classes['states']
        self.common_data['node_ids'] = [n['id'] for n in classes['nodes']]
        self.common_data['node_idx_from_id'] = {int(n['id']):i for i,n in enumerate(classes['nodes'])}
        
        self.state_features = {}
        self.common_data['feature_order'] = ['class', 'category']
        for loc, (role, values) in enumerate(ROLES.items()):
            self.common_data['feature_order'].append(role)
            for idx, v in enumerate(values):
                self.state_features[v] = np.zeros((len(ROLES)))
                self.state_features[v][loc] = idx + 1
            self.state_features["NONE"] = np.zeros((len(ROLES)))

        self.common_data['node_roles'] = ROLES
        self.common_data['node_roles']['class'] = list(set(self.common_data['node_classes']))
        self.common_data['node_roles']['class'].sort()
        self.common_data['node_roles']['category'] = list(set(self.common_data['node_categories']))
        self.common_data['node_roles']['category'].sort()
        self.common_data['feature_options_num'] = [len(self.common_data['node_roles'][f]) for f in self.common_data['feature_order']]

        # Diagonal nodes are always irrelevant
        self.nonstatic_edges = 1 - np.eye(len(classes['nodes']))
        # Rooms, furniture and appliances nodes don't move
        node_categories = [n['category'] for n in classes['nodes']]
        self.nonstatic_edges[np.where(np.array(node_categories) == "Rooms"),:] = 0
        self.nonstatic_edges[np.where(np.array(node_categories) == "Furniture"),:] = 0
        self.nonstatic_edges[np.where(np.array(node_categories) == "Decor"),:] = 0
        self.nonstatic_edges[np.where(np.array(node_categories) == "Appliances"),:] = 0
        print('Number of nonstatic edges = ',sum(self.nonstatic_edges.max(axis=1)))
        self.seen_edges = np.zeros_like(self.nonstatic_edges)

        self.home_graph = None

        self.common_data['edge_keys'] = classes['edges']
        static = lambda category : category in ["Furniture", "Room"]
        self.common_data['static_nodes'] = [n['id'] for n in classes['nodes'] if static(n['category'])]


    def read_data(self, data_dir, output_dir):
        idx_map = []
        for root, dirs, files in os.walk(data_dir):
            assert dirs == []
            if INTERACTIVE:
                inp = input(f'Do you want to visualize all {len(files)} routines?')
            else:
                inp = 'n'
            viz = (inp == 'y')
            for f in files:
                with open(os.path.join(root,f)) as f_in:
                    routine = json.load(f_in)
                nodes, edges = self.read_graphs(routine["graphs"])
                if viz:
                    visualize_raw_and_parsed_routine(routine, edges, nodes, self.common_data['node_classes'], self.common_data['node_categories'])
                    inp = input(f'Do you want to visualize the next routine?')
                    viz = (inp == 'y')
                self.home_graph = edges[0]
                times = torch.Tensor(routine["times"])
                file_basename = os.path.splitext(f)[0]
                f_out = os.path.join(output_dir,file_basename+'.pt')
                samples = self.make_pairwise(nodes, edges, times)
                for i in range(len(samples)):
                    idx_map.append((file_basename, i))
                torch.save(samples, f_out)

        self.seen_edges[self.nonstatic_edges == 0] = 0
        self.seen_edges = torch.Tensor(self.seen_edges)
        self.nonstatic_edges = torch.Tensor(self.nonstatic_edges)

        return idx_map
        

    def read_graphs(self, graphs):
        num_nodes = len(self.common_data['node_ids'])
        node_features_list = []
        edge_features_list = []
        for i,graph in enumerate(graphs):
            assert len(graph['nodes']) == len(self.common_data['node_idx_from_id']), "Weren't we supposed to have the same number of nodes each time?!"
            node_features = np.zeros((len(graph['nodes']),self.feature_from_node().shape[-1]))
            for node in graph['nodes']:
                node_features[self.common_data['node_idx_from_id'][node['id']],:] = self.feature_from_node(node)
            node_features_list.append(node_features)
            edge_features = np.zeros((num_nodes, num_nodes))
            for e in graph['edges']:
                if e['relation_type'] in self.common_data['edge_keys']:
                    edge_features[self.common_data['node_idx_from_id'][e['from_id']],self.common_data['node_idx_from_id'][e['to_id']]] = 1
            original_edges = edge_features[:,:]
            edge_features[:,:] = _sparsify(edge_features[:,:])
            if (edge_features[:,:].sum(axis=-1)).max() != 1:
                print(f"Matrix {i} not really a tree \n{edge_features[:,:]}")
                not_a_tree(original_edges, edge_features[:,:], self.common_data['node_classes'])
            assert (edge_features[:,:].sum(axis=-1)).max() == 1, f"Matrix {i} not really a tree \n{edge_features[:,:]}"
            self.seen_edges[:,:] += edge_features[:,:]
            edge_features_list.append(edge_features)
        return torch.Tensor(np.array(node_features_list)), torch.Tensor(np.array(edge_features_list))

    def make_pairwise(self, nodes, edges, times):
        pairwise_samples = []
        self.time_min = torch.Tensor([float("Inf")])
        self.time_max = -torch.Tensor([float("Inf")])
    
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
            # 3 = to home state; 1 = from home state; 2 = neither; 0 = no change
            if prev_edges is not None:
                change_type = (np.absolute(edges[data_idx] - prev_edges)).sum(-1)
                change_type += (self.home_graph * (edges[data_idx] - prev_edges)).sum(-1)
                pairwise_samples.append({'prev_edges': torch.Tensor(prev_edges), 
                                         'prev_nodes': prev_nodes, 
                                         'time': t, 
                                         'edges': torch.Tensor((edges[data_idx])), 
                                         'nodes': nodes[data_idx], 
                                         'change_type': torch.Tensor(change_type).to(int)})
                # if (prev_nodes != nodes[data_idx]).max() > 0:
                #     print(prev_nodes)
                #     print(nodes[data_idx])
                #     print(prev_nodes - nodes[data_idx])
                #     print('\n\n')
            prev_edges = edges[data_idx]
            prev_nodes = nodes[data_idx]
        return pairwise_samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model on routines.')
    parser.add_argument('--path', type=str, default='data/0720_stretched/0720_stretched_breakfast', help='Path where the data lives. Must contain routines, info and classes json files.')
    args = parser.parse_args()

    data_path = (os.path.join(args.path, 'routines_train'), os.path.join(args.path, 'routines_test'))
    if not (os.path.exists(data_path[0]) and os.path.exists(data_path[1])):
        print('The data directory must contain both routines_train and routines_test directories')
    classes_path = os.path.join(args.path, 'classes.json')
    info_path = os.path.join(args.path, 'info.json')

    ProcessDataset(data_path=data_path, classes_path=classes_path, info_path=info_path, output_path=os.path.join(args.path, 'processed'))