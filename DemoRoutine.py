# %% [markdown]
# ### Prerequisites

# %%
import os
import shutil
import glob
import numpy as np
import networkx as nx
from copy import deepcopy
import random
import json

import torch
from GraphTranslatorModule import GraphTranslatorModule
from encoders import TimeEncodingOptions
from graph_visualizations import visualize_conditional_datapoint
from readerFileBased import RoutinesDataset


# %%
class Graph():
    def __init__(self, root='kitchen'):
        self.root = root
        self.parents = {}
        self.category = {root:'Rooms'}
    
    def add(self, node, parent, category='placable_object'):
        assert parent in self.parents.keys() or parent == self.root
        self.parents[node] = parent
        self.category[node] = category

    def draw(self):
        G = nx.DiGraph()
        G.add_nodes_from(list(self.parents.keys()))
        G.add_edges_from([(n, self.parents[n]) for n in self.parents])
        still_remaining = deepcopy(list(self.parents.keys()))
        nodelists = [[self.root]]
        while(len(still_remaining)):
            new_layer = []
            remaining_nodes = still_remaining
            still_remaining = []
            for n in remaining_nodes:
                if self.parents[n] in nodelists[-1]:
                    # print(n)
                    new_layer.append(n)
                else:
                    still_remaining.append(n)
            nodelists.append(new_layer)
            # print(new_layer)
        nx.draw_networkx(G)
    
    def move(self, object, location):
        assert location in self.parents.keys() or location == self.root
        self.parents[object] = location
    
    def to_dict(self):
        id_map = {self.root:0}
        id_map .update({n:i+1 for i,n in enumerate(self.parents.keys())})
        graph_dict = {}
        graph_dict['nodes'] = [{'id':i, 'class_name':n, 'category':self.category[n]} for n,i in id_map.items()]
        graph_dict['edges'] = [{'from_id':id_map[n], 'to_id':id_map[p], 'relation_type':'INSIDE'} for n,p in self.parents.items()]
        return graph_dict

    def dump(self, filename):
        graph_dict = self.to_dict()
        with open (filename, 'w') as f:
            json.dump(graph_dict, f, indent=4)

    def num_nodes(self):
        return len(self.parents) + 1
    
    def get_edges(self, node_name, node_id):
        edges = np.zeros((len(node_name), len(node_name)))
        for i,node in enumerate(node_name):
            if node != self.root:
                edges[i,node_id[self.parents[node]]] = 1
        return torch.Tensor(edges).unsqueeze(0)


# %%
def print_changes(edges1, edges_list, node_name, code=False):
    new_locs = {}
    for edges2 in edges_list:
        loc1 = edges1.argmax(-1).reshape(-1)
        loc2 = edges2.argmax(-1).reshape(-1)
        for i,(l1, l2) in enumerate(zip(loc1, loc2)):
            if l1 != 0 and l1 != l2:
                if node_name[i] not in new_locs:
                    new_locs[node_name[i]] = node_name[l2]
                    if code:
                        print(f"graph.move('{node_name[i]}', '{node_name[l2]}')")
                    else:
                        print(f'Object {node_name[i]} moved from {node_name[l1]} to {node_name[l2]}')
    print('time += 20')
# %%
def d(dmin, dmax):
    return (random.random()*(dmax-dmin))+dmin

# %% [markdown]
# ### Generating data

# %%
def get_initial_graph(draw=False):
    initial_graph = Graph('kitchen')
    initial_graph.add(node='shelf', parent='kitchen', category='Furniture')
    initial_graph.add(node='drawer', parent='kitchen', category='Furniture')
    initial_graph.add(node='sink', parent='kitchen', category='Furniture')
    initial_graph.add(node='counter', parent='kitchen', category='Furniture')
    initial_graph.add(node='fridge', parent='kitchen', category='Furniture')
    initial_graph.add(node='table', parent='kitchen', category='Furniture')

    initial_graph.add(node='cereal', parent='shelf')
    initial_graph.add(node='bowl', parent='drawer')
    initial_graph.add(node='pills', parent='drawer')
    initial_graph.add(node='banana', parent='counter')
    initial_graph.add(node='glass', parent='counter')
    initial_graph.add(node='milk', parent='fridge')

    if draw:
        initial_graph.draw()

    return initial_graph

# %%
def execute_routine(graph, initial_time, steps):
    time = initial_time
    graph_list = [deepcopy(graph.to_dict())]
    time_list = [initial_time]
    for ((obj, loc), (dmin, dmax)) in steps:
        duration = d(dmin, dmax)
        graph.move(obj, loc)
        time += duration
        graph_list.append(deepcopy(graph.to_dict()))
        time_list.append(deepcopy(time))
    obj_in_use = [[] for _ in range(len(graph_list))]
    return {"graphs":graph_list, "times": time_list, "objects_in_use":obj_in_use}


breakfast_routine_steps = [
        (('cereal', 'table'), (60,90)),
        (('milk', 'table'), (4,6)),
        (('bowl', 'table'), (4,6)),
        (('glass', 'table'), (4,6)),
        (('milk', 'fridge'), (20,30)),
        (('bowl', 'sink'), (20,30)),
        (('glass', 'sink'), (4,6)),
        (('cereal', 'shelf'), (4,6))
    ]
    
medication_steps = [
        (('pills', 'table'), (90,120)),
        (('glass', 'table'), (4,6)),
        (('glass', 'sink'), (20,30)),
        (('pills', 'drawer'), (4,6))
]

# %%
data_directory = 'data/demo_routines'
logs_directory = 'logs/demo_routines'

def generate_data():
    if os.path.exists(data_directory):
        shutil.rmtree(data_directory)

    initial_graph = get_initial_graph()
    os.makedirs(data_directory)

    inf = {"dt": 5, 
    "num_train_routines": 50, 
    "num_test_routines": 10, 
    "weekend_days": [], 
    "start_time": 0, 
    "end_time": 210, 
    "interleaving": False, 
    "only_used_objects": True, 
    "num_nodes": initial_graph.num_nodes(), 
    "search_object_ids": [], 
    "search_object_names": []}

    with open(os.path.join(data_directory,"info.json"), 'w') as f:
        json.dump(inf,f)
    with open(os.path.join(data_directory,"classes.json"), 'w') as f:
        classes_dict = {'nodes':initial_graph.to_dict()['nodes'], 'edges':['INSIDE','ON']}
        json.dump(classes_dict,f)

    os.makedirs(os.path.join(data_directory,'routines_train'))
    for i in range(inf["num_train_routines"]):
        routine = execute_routine(get_initial_graph(), inf["start_time"], breakfast_routine_steps+medication_steps)
        
        with open(os.path.join(data_directory,'routines_train',"{:3d}.json".format(i)), 'w') as f:
            json.dump(routine, f)

    os.makedirs(os.path.join(data_directory,'routines_test'))
    for i in range(inf["num_test_routines"]):
        routine = execute_routine(get_initial_graph(), inf["start_time"], breakfast_routine_steps+medication_steps)
        
        with open(os.path.join(data_directory,'routines_test',"{:3d}.json".format(i)), 'w') as f:
            json.dump(routine, f)



def evaluate_model(graph, time):
    with open(data_directory+'/processed/common_data.json') as f:
        common_data = json.load(f)
    log_dir = logs_directory+'/ours_5epochs'

    node_name = common_data['node_classes']
    node_id = {n:i for i,n in enumerate(node_name)}
    standard_nodes = torch.Tensor(np.eye(len(node_name))).unsqueeze(0)
    with open (os.path.join(log_dir,'config.json')) as f:
        cfg = json.load(f)

    time_encoding_func = TimeEncodingOptions()(cfg['TIME_ENCODING'])
    data=cfg['DATA_PARAM']
    checkpoint_file = glob.glob(log_dir+'/*.ckpt')[0]

    model = GraphTranslatorModule.load_from_checkpoint(checkpoint_file, 
                                                        num_nodes=data['n_nodes'],
                                                        node_feature_len=data['n_len'],
                                                        context_len=data['c_len'],
                                                        edge_importance=cfg['EDGE_IMPORTANCE'],
                                                        edge_dropout_prob = cfg['EDGE_DROPOUT_PROB'],
                                                        tn_loss_weight=cfg['TN_LOSS_WEIGHT'],
                                                        learned_time_periods=cfg['LEARNED_TIME_PERIODS'],
                                                        hidden_layer_size=cfg['HIDDEN_LAYER_SIZE'])

    model.load_state_dict(torch.load(os.path.join(log_dir,'weights.pt')))

    proactivity_window = 5

    prev_edges = graph.get_edges(node_name,node_id)

    batch = {}

    batch['edges'] = prev_edges
    batch['nodes'] = standard_nodes
    batch['y_edges'] = prev_edges
    batch['y_nodes'] = standard_nodes
    batch['dynamic_edges_mask'] = torch.tensor(np.ones_like(prev_edges)-np.eye((prev_edges.size()[-1])))
    batch['dynamic_edges_mask'][:,:7,:] = 0
    batch['context'] = time_encoding_func(time)

    edges_seq = []

    for _ in range(proactivity_window):
        _, details = model.step(batch)
        edg = (details['output_probs']['location']).to(torch.float32)
        time += cfg['DATA_INFO']['dt']
        edges_seq.append(edg)
        batch['edges'] = edg
        batch['context'] = time_encoding_func(time)
        
    print_changes(prev_edges, edges_seq, node_name, code=True)

    # data = RoutinesDataset(data_path=os.path.join(data_directory,'processed'), 
    #                         time_encoder=time_encoding_func, 
    #                         batch_size=cfg['BATCH_SIZE'],
    #                         only_seen_edges = cfg['ONLY_SEEN_EDGES'],
    #                         max_routines = (50, None))

    # visualize_conditional_datapoint(model, data.get_single_example_test_loader(), data.node_classes, use_output_nodes=cfg['LEARN_NODES'])
           
