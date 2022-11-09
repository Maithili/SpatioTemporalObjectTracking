import json
import os
import shutil
import argparse
import numpy as np
from math import ceil, floor
import torch
from encoders import time_external
from torch.utils.data import DataLoader
from breakdown_evaluations import activity_list

# from graph_visualizations import visualize_routine, visualize_parsed_routine

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


# def get_cooccurence_frequency(trainset):
#     # one-smoothing
#     all_edges = torch.ones_like(trainset[0][0])
#     for r in trainset:
#         all_edges = all_edges + r[0]
#     prior = all_edges/(all_edges.sum(dim=0))/(len(trainset) + 1)
#     print('Size of coccurence prior : ',prior.size())
#     return prior

# def get_spectral_components(trainset, periods_mins):
#     reals = [torch.zeros_like(trainset[0][0]) for _ in periods_mins]
#     imags = [torch.zeros_like(trainset[0][0]) for _ in periods_mins]
#     for idx in range(len(trainset)):
#         edges, time = trainset.get_edges_and_time(idx)
#         for harmonic, period in enumerate(periods_mins):
#             reals[harmonic] += edges * np.cos(2*np.pi*time/period)
#             imags[harmonic] += edges * np.sin(2*np.pi*time/period)
#     components = []
#     for r,i,p in zip(reals, imags, periods_mins):
#         components.append({'amplitude': (np.sqrt(np.square(r)+np.square(i))/len(trainset)), 'phase': (np.arctan2(i,r)), 'period': p})
#     print('Size of FreMeN prior : ',str([(comp['amplitude'].size(),comp['phase'].size()) for comp in components]))
#     return components



class OneHotEmbedder():
    def __init__(self):
        pass

    def get_func(self, class_list):
        return lambda idxs: torch.nn.functional.one_hot(idxs.to(int))

class ConceptNetEmbedder():

    def __init__(self, file='helpers/numberbatch-en.txt'):
        '''
        Loads Conceptnet Numberbatch from the text file
        Args:
            file: path to numberbatch_en.txt file (must be on local system)
        Output:
            embeddings_index: dictionary mapping objects to their numberbatch embbs
            num_feats: length of numberbatch embedding
        '''

        # Create dictionary of object: vector
        self.embeddings_index = dict()

        with open(file, 'r', encoding="utf8") as f:

            # Parse text file to populate dictionary
            for line in f:
                values = line.split()
                word = values[0]

                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefs

        self.num_feats = len(coefs)

        print('ConceptNet loaded')

        self.synonyms = {
            'tvstand': 'tv_stand',
            'cookingpot': 'cooking_pot',
            'knifeblock': 'knife_block',
        }

    def __call__(self, token):
        token = token.lower()
        if token in self.synonyms: token = self.synonyms[token]
        if token in self.embeddings_index:
            assert len(self.embeddings_index[token]) == self.num_feats
            return self.embeddings_index[token]
        elif '_' in token:
            subtokens = token.split('_')
            try:
                assert all([len(self.embeddings_index[t]) == self.num_feats for t in subtokens])
                return sum([self.embeddings_index[t] for t in subtokens])
            except:
                raise KeyError(f'{subtokens} cannot be embedded through ConceptNet')
        raise KeyError(f'{token} cannot be embedded through ConceptNet')

    def get_func(self, class_list):
        embedding_list = [self(cls) for cls in class_list]
        embedding_tensor = torch.Tensor(np.array(embedding_list))
        embedding_func = lambda idxs : torch.matmul(torch.nn.functional.one_hot(idxs.to(int)).float(), embedding_tensor.float())
        return embedding_func


class CollateToDict():
    def __init__(self, dict_labels):
        self.dict_labels = dict_labels

    def __call__(self, tensor_tuple, remove=None):
        data = {label:torch.Tensor() for label in self.dict_labels}
        if len(tensor_tuple) > 1:
            max_sequence_len = max([tensor[0] for tensor in tensor_tuple])
            assert remove is None, "'remove' argument not supported for batches >1"
        else:
            max_sequence_len = tensor_tuple[0][0]
        def pad_tensor(t):
            zeros_size = list(t.size())
            zeros_size[0] = max_sequence_len - t.size(0)
            t = torch.cat([t,torch.zeros(torch.Size(zeros_size))], dim=0)
            return t.unsqueeze(0)
        for i,label in enumerate(self.dict_labels):
            if remove is not None:
                data[label] = pad_tensor(tensor_tuple[0][i+1][remove[0]:remove[1]])
            else:
                data[label] = torch.cat([pad_tensor(tensors[i+1]) for tensors in tensor_tuple], dim=0)
        return data

class DataSplit():
    def __init__(self, routines_dir, time_encoder, active_edges, max_num_files=None, node_embedder=OneHotEmbedder):
        self.time_encoder = time_encoder
        self.active_edges = active_edges
        self.routines_dir = routines_dir
        self.collate_fn = CollateToDict(['edges', 'nodes', 'activity', 'time', 'context_time', 'dynamic_edges_mask'])
        self.files = [name for name in os.listdir(self.routines_dir) if os.path.isfile(os.path.join(self.routines_dir, name))]
        self.files.sort()
        self.max_num_files = min(max_num_files,len(self.files)) if max_num_files is not None else len(self.files)
        self.node_embedder = node_embedder

    def __len__(self):
        return self.max_num_files

    def __getitem__(self, idx: int):
        data = torch.load(os.path.join(self.routines_dir, self.files[idx]))
        stacked_edges, stacked_nodes, stacked_activities, stacked_times = data
        return stacked_times.size()[0], stacked_edges, self.node_embedder(stacked_nodes), stacked_activities, stacked_times, self.time_encoder(stacked_times), self.active_edges.unsqueeze(0).repeat(stacked_times.size()[0],1,1)


class RoutinesDataset():
    def __init__(self, data_path, 
                 time_encoder = time_external, 
                 batch_size = 1,
                 max_routines = (None, None),
                 use_conceptnet = True):

        with open(os.path.join(data_path, 'common_data.json')) as f:
            self.common_data = json.load(f)

        self.time_encoder = time_encoder
        
        self.params = {}
        self.params['dt'] = self.common_data['info']['dt']
        self.params['batch_size'] = batch_size

        self.active_edges = torch.load(os.path.join(data_path, 'nonstatic_edges.pt')) #, map_location=lambda storage, loc: storage.cuda(1))
        
        self.home_graph = torch.load(os.path.join(data_path, 'home_graph.pt'))

        self.node_ids = self.common_data['node_ids']
        self.node_classes = self.common_data['node_classes']
        self.node_categories = self.common_data['node_categories']
        self.node_idx_from_id = {int(k):v for k,v in self.common_data['node_idx_from_id'].items()}
        self.edge_keys = self.common_data['edge_keys']
        self.static_nodes = self.common_data['static_nodes']

        if use_conceptnet:
            node_embedder = ConceptNetEmbedder()
        else:
            node_embedder = OneHotEmbedder()
        
            
        # Generate train and test loaders
        self.train = DataSplit(os.path.join(data_path,'train'), self.time_encoder, self.active_edges, max_num_files=max_routines[0], node_embedder=node_embedder.get_func(self.node_classes))
        self.test = DataSplit(os.path.join(data_path,'test'), self.time_encoder, self.active_edges, max_num_files=max_routines[1], node_embedder=node_embedder.get_func(self.node_classes))
        print('Train split has ',len(self.train),' routines')
        print('Test split has ',len(self.test),' routines')

        # Infer parameters for the model
        model_data = self.test.collate_fn([self.test[0]])
        self.params['n_nodes'] = model_data['nodes'].size()[-2]
        self.params['n_len'] = model_data['nodes'].size()[-1]
        self.params['c_len'] = model_data['context_time'].size()[-1]
        print(self.common_data['activities'], len(self.common_data['activities']))
        self.params['n_activities'] = len(self.common_data['activities'])
        self.params['null_activity_idx'] = self.common_data['activities'].index(None)

    def get_train_loader(self):
        return DataLoader(self.train, num_workers=min(4,os.cpu_count()), batch_size=self.params['batch_size'], collate_fn=self.train.collate_fn)

    def get_test_loader(self):
        return DataLoader(self.test, num_workers=min(4,os.cpu_count()), batch_size=self.params['batch_size'], collate_fn=self.test.collate_fn)

    def get_test_split(self):
        return self.test

    def get_single_example_test_loader(self):
        return DataLoader(self.test, num_workers=min(4,os.cpu_count()), batch_size=1, collate_fn=self.test.collate_fn)
    
    # def get_cooccurence_frequency(self):
    #     return get_cooccurence_frequency(self.train)

    # def get_spectral_components(self, periods_mins):
    #     return get_spectral_components(self.train, periods_mins)



class ProcessDataset():
    def __init__(self, data_path, 
                 classes_path,
                 info_path, 
                 output_path):

        output_train_path = os.path.join(output_path, 'train')
        output_test_path = os.path.join(output_path, 'test')
        if os.path.exists(output_train_path): 
            overwrite = input(f'Dataset seems to already exist at {output_path}!! Overwrite? (y/n)')
            if overwrite != 'y': raise InterruptedError('Cancelling data processing...')
            shutil.rmtree(output_path)
        os.makedirs(output_train_path)
        if os.path.exists(output_test_path): 
            overwrite = input(f'Dataset seems to already exist at {output_path}!! Overwrite? (y/n)')
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
        with open(os.path.join(output_path, 'common_data.json'), 'w') as f:
            json.dump(self.common_data, f)
        torch.save(self.seen_edges, os.path.join(output_path, 'seen_edges.pt'))
        torch.save(self.nonstatic_edges, os.path.join(output_path, 'nonstatic_edges.pt'))
        torch.save(self.home_graph, os.path.join(output_path, 'home_graph.pt'))

    def read_classes(self, classes_path):
        with open(classes_path, 'r') as f:
            classes = json.load(f)
        def ignore_node(node):
            return node['class_name'].startswith('clothes_')
        self.common_data['node_ids'] = [n['id'] for n in classes['nodes'] if not ignore_node(n)]
        self.common_data['node_classes'] = [n['class_name'] for n in classes['nodes'] if not ignore_node(n)]
        self.common_data['node_categories'] = [n['category'] for n in classes['nodes'] if not ignore_node(n)]
        self.common_data['node_idx_from_id'] = {int(n):i for i,n in enumerate(self.common_data['node_ids'])}
        self.common_data['activities'] = activity_list
        self.common_data['actions'] = []

        # Diagonal nodes are always irrelevant
        self.nonstatic_edges = 1 - np.eye(len(self.common_data['node_ids']))
        # Rooms, furniture and appliances nodes don't move
        self.nonstatic_edges[np.where(np.array(self.common_data['node_categories']) == "Rooms"),:] = 0
        self.nonstatic_edges[np.where(np.array(self.common_data['node_categories']) == "Furniture"),:] = 0
        self.nonstatic_edges[np.where(np.array(self.common_data['node_categories']) == "Decor"),:] = 0
        self.nonstatic_edges[np.where(np.array(self.common_data['node_categories']) == "Appliances"),:] = 0
        self.seen_edges = np.zeros_like(self.nonstatic_edges)

        self.home_graph = None

        self.common_data['edge_keys'] = classes['edges']
        static = lambda category : category in ["Furniture", "Room"]
        self.common_data['static_nodes'] = [n['id'] for n in classes['nodes'] if static(n['category']) and not ignore_node(n)]


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
                fpath = os.path.join(root,f)
                with open(fpath) as f_in:
                    routine = json.load(f_in)
                nodes, edges = self.read_graphs(routine["graphs"])
                # actions = self.read_actions(routine["actions"])
                # activities =self.read_activities(routine["activity"])
                # if viz:
                #     visualize_routine(routine)
                #     visualize_parsed_routine(edges, nodes, self.common_data['node_classes'])
                #     inp = input(f'Do you want to visualize the next routine?')
                #     viz = (inp == 'y')
                self.home_graph = edges[0,:,:]
                times = torch.Tensor(routine["times"])
                file_basename = os.path.splitext(f)[0]
                f_out = os.path.join(output_dir,file_basename+'.pt')
                activity_func = self.activity_from_time(fpath.replace('routines','scripts').replace('json','txt'))
                stacked_edges, stacked_nodes, stacked_activities, stacked_times = self.stack_routine(nodes, edges, times, activity_func)
                idx_map.append((file_basename, len(stacked_times)))
                torch.save([stacked_edges, stacked_nodes, stacked_activities, stacked_times], f_out)

        self.seen_edges[self.nonstatic_edges == 0] = 0
        self.seen_edges = torch.Tensor(self.seen_edges)
        self.nonstatic_edges = torch.Tensor(self.nonstatic_edges)

        return idx_map
        
    def activity_from_time(self, script_file):
        scr_header_lines = open(script_file).read().split('\n\n\n')[0].split('\n')
        def parse_time(ts):
            parts = [int(t) for t in ts.split(':')]
            if len(parts) == 2:
                return parts[0]*60 + parts[1]
            if len(parts) == 3:
                return (parts[0]*24 + parts[1])*60 + parts[2]
            raise RuntimeError()
        def parse_line(l):
            activity = l[:l.index('(')-1].strip()
            timerange = l[l.index('(')+1: l.index(')')]
            timerange = timerange.replace('1day - ','01:')
            start_time = parse_time(timerange.split('-')[0].strip())
            end_time = parse_time(timerange.split('-')[1].strip())
            return activity, lambda t: t>=start_time and t<end_time
        activities = {parse_line(l)[0]:parse_line(l)[1] for l in scr_header_lines}
        def activity_func(t):
            options = [self.common_data['activities'].index(a) for a,fun in activities.items() if fun(t)]
            if len(options) > 0:
                return options[0]
            else:
                return self.common_data['activities'].index(None)
        return activity_func

    def read_graphs(self, graphs):
        num_nodes = len(self.common_data['node_ids'])
        node_features = np.zeros((len(graphs), num_nodes))
        edge_features = np.zeros((len(graphs), num_nodes, num_nodes))
        for i,graph in enumerate(graphs):
            node_features[i,:] = np.arange(num_nodes)
            for e in graph['edges']:
                if e['relation_type'] in self.common_data['edge_keys'] and e['from_id'] in self.common_data['node_ids'] and e['to_id'] in self.common_data['node_ids']:
                    edge_features[i,self.common_data['node_idx_from_id'][e['from_id']],self.common_data['node_idx_from_id'][e['to_id']]] = 1
            original_edges = edge_features[i,:,:]
            edge_features[i,:,:] = _sparsify(edge_features[i,:,:])
            if (edge_features[i,:,:].sum(axis=-1)).max() != 1:
                print(f"Matrix {i} not really a tree \n{edge_features[i,:,:]}")
                not_a_tree(original_edges, edge_features[i,:,:], self.common_data['node_classes'])
            assert (edge_features[i,:,:].sum(axis=-1)).max() == 1, f"Matrix {i} not really a tree \n{edge_features[i,:,:]}"
            self.seen_edges[:,:] += edge_features[i,:,:]
        return torch.Tensor(node_features), torch.Tensor(edge_features)

    # def read_actions(self, actions_list):
    #     encoded_action_list = []
    #     for actions in actions_list:
    #         encoded_action_list.append([])
    #         for i,action in enumerate(actions):
    #             if action[0] not in self.common_data['actions']:
    #                 self.common_data['actions'].append(action[0])
    #             action_idx = self.common_data['actions'].index(action[0])
    #             new_action = [(i, 0, action_idx)]
    #             if len(action) > 1:
    #                 for a in action[1:]:
    #                     if a not in self.common_data['node_idx_from_id'].keys():
    #                         print(f'Object {a} in {action} not found')
    #                         continue
    #                     new_action.append((i, 1, self.common_data['node_idx_from_id'][a]))
    #             encoded_action_list[-1] += new_action
    #     return encoded_action_list

    # def read_activities(self, activities_list):
    #     for activity in activities_list:
    #         if activity not in self.common_data['activities']:
    #             self.common_data['activities'].append(activity)
    #     return torch.Tensor([self.common_data['activities'].index(activity) for activity in activities_list]).to(int)
    

    def stack_routine(self, nodes, edges, times, activity_func):
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
        all_nodes = []
        all_edges = []
        all_activities = []
        all_times = []
        for t in range(time_min, time_max, self.dt):
            while t >= times[data_idx+1]:
                data_idx += 1
            if data_idx < 0:
                continue
            all_edges.append(edges[data_idx].unsqueeze(0))
            all_nodes.append(nodes[data_idx].unsqueeze(0))
            all_activities.append(torch.Tensor([activity_func(t)]))
            all_times.append(torch.Tensor([t]))
        stacked_edges = torch.cat(all_edges)
        stacked_nodes = torch.cat(all_nodes)
        stacked_activities = torch.cat(all_activities)
        stacked_times = torch.cat(all_times)
        return stacked_edges, stacked_nodes, stacked_activities, stacked_times


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model on routines.')
    parser.add_argument('--path', type=str, default='data/personaWithoutClothesAllObj/persona0', help='Path where the data lives. Must contain routines, info and classes json files.')
    args = parser.parse_args()


    data_path = (os.path.join(args.path, 'routines_train'), os.path.join(args.path, 'routines_test'))
    if not (os.path.exists(data_path[0]) and os.path.exists(data_path[1])):
        print('The data directory must contain both routines_train and routines_test directories')
    classes_path = os.path.join(args.path, 'classes.json')
    info_path = os.path.join(args.path, 'info.json')

    ProcessDataset(data_path=data_path, classes_path=classes_path, info_path=info_path, output_path=os.path.join(args.path, 'processed_seq'))