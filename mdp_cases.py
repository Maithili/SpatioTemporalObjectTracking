#! ./.venv/bin/python

import os
import json
import yaml
import shutil
from random import random

from matplotlib.pyplot import get
from run import run, DEFAULT_CONFIG

node_dictionary = {}
node_dictionary['kitchen'] = {"id": 1, "class_name": "kitchen", "category": "Rooms", "properties": [], "states": [], "prefab_name": None, "bounding_box": None}

node_dictionary['cabinet'] = {"id": 2, "class_name": "cabinet", "category": "Furniture", "properties": [], "states": ["CLOSED"], "prefab_name": None, "bounding_box": None}
node_dictionary['counter'] = {"id": 3, "class_name": "counter", "category": "Furniture", "properties": [], "states": ["CLEAN"], "prefab_name": None, "bounding_box": None}
node_dictionary['table'] = {"id": 4, "class_name": "table", "category": "Furniture", "properties": [], "states": ["CLOSED"], "prefab_name": None, "bounding_box": None}
node_dictionary['fridge'] = {"id": 5, "class_name": "table", "category": "Furniture", "properties": [], "states": ["CLOSED"], "prefab_name": None, "bounding_box": None}
node_dictionary['sink'] = {"id": 6, "class_name": "sink", "category": "Furniture", "properties": [], "states": ["CLOSED"], "prefab_name": None, "bounding_box": None}

node_dictionary['cup'] = {"id": 7, "class_name": "cup", "category": "placable_objects", "properties": [], "states": ["CLEAN"], "prefab_name": None, "bounding_box": None}
node_dictionary['plate'] = {"id": 8, "class_name": "plate", "category": "placable_objects", "properties": [], "states": ["CLEAN"], "prefab_name": None, "bounding_box": None}
node_dictionary['bowl'] = {"id": 9, "class_name": "bowl", "category": "placable_objects", "properties": [], "states": ["CLEAN"], "prefab_name": None, "bounding_box": None}
node_dictionary['spoon'] = {"id": 10, "class_name": "spoon", "category": "placable_objects", "properties": [], "states": ["CLEAN"], "prefab_name": None, "bounding_box": None}
node_dictionary['pan'] = {"id": 11, "class_name": "pan", "category": "placable_objects", "properties": [], "states": ["CLEAN"], "prefab_name": None, "bounding_box": None}

node_dictionary['cereal'] = {"id": 12, "class_name": "cereal", "category": "placable_objects", "properties": [], "states": ["CLEAN"], "prefab_name": None, "bounding_box": None}
node_dictionary['coffee'] = {"id": 13, "class_name": "coffee", "category": "placable_objects", "properties": [], "states": ["CLEAN"], "prefab_name": None, "bounding_box": None}
node_dictionary['butter'] = {"id": 14, "class_name": "butter", "category": "placable_objects", "properties": [], "states": ["CLEAN"], "prefab_name": None, "bounding_box": None}
node_dictionary['cheese'] = {"id": 15, "class_name": "cheese", "category": "placable_objects", "properties": [], "states": ["CLEAN"], "prefab_name": None, "bounding_box": None}
node_dictionary['bread'] = {"id": 16, "class_name": "bread", "category": "placable_objects", "properties": [], "states": ["CLEAN"], "prefab_name": None, "bounding_box": None}
node_dictionary['milk'] = {"id": 17, "class_name": "milk", "category": "placable_objects", "properties": [], "states": ["CLEAN"], "prefab_name": None, "bounding_box": None}
node_dictionary['eggs'] = {"id": 18, "class_name": "eggs", "category": "placable_objects", "properties": [], "states": ["CLEAN"], "prefab_name": None, "bounding_box": None}
node_dictionary['apple'] = {"id": 19, "class_name": "apple", "category": "placable_objects", "properties": [], "states": ["CLEAN"], "prefab_name": None, "bounding_box": None}
node_dictionary['chocolate'] = {"id": 20, "class_name": "chocolate", "category": "placable_objects", "properties": [], "states": ["CLEAN"], "prefab_name": None, "bounding_box": None}

def edge(from_node, relation, to_node):
    return {'from_id':node_dictionary[from_node]['id'], 'relation_type':relation, 'to_id':node_dictionary[to_node]['id']}

def time_mins(mins, hrs, days=0, weeks=0):
    return int(round(((((weeks*7+days)*24)+hrs)*60+mins)))


def get_edges(time_idx = 0, type_idx = 1):

    time = ['before','prep','after','cleaned'][time_idx]
    type = ['eggs','cereal','bread'][type_idx]

    edges = []

    edges.append(edge('cabinet','INSIDE','kitchen'))
    edges.append(edge('counter','INSIDE','kitchen'))
    edges.append(edge('table','INSIDE','kitchen'))
    edges.append(edge('fridge','INSIDE','kitchen'))
    edges.append(edge('sink','INSIDE','kitchen'))

    if time == 'before' or time == 'after':
        edges.append(edge('cup','INSIDE','cabinet'))
        edges.append(edge('plate','INSIDE','cabinet'))
        edges.append(edge('bowl','INSIDE','cabinet'))
        edges.append(edge('spoon','INSIDE','cabinet'))
        edges.append(edge('pan','INSIDE','cabinet'))

        edges.append(edge('cereal','INSIDE','cabinet'))
        edges.append(edge('coffee','INSIDE','cabinet'))
        edges.append(edge('butter','INSIDE','fridge'))
        edges.append(edge('cheese','INSIDE','fridge'))
        edges.append(edge('bread','INSIDE','fridge'))
        edges.append(edge('milk','INSIDE','fridge'))
        edges.append(edge('eggs','INSIDE','fridge'))
        edges.append(edge('apple','ON','counter'))
        edges.append(edge('chocolate','ON','counter'))

    elif type == 'eggs':
        edges.append(edge('bowl','INSIDE','cabinet'))
        edges.append(edge('spoon','INSIDE','cabinet'))

        edges.append(edge('cereal','INSIDE','cabinet'))
        edges.append(edge('butter','INSIDE','fridge'))
        edges.append(edge('milk','INSIDE','fridge'))
        edges.append(edge('apple','ON','counter'))
        edges.append(edge('chocolate','ON','counter'))
        
        if time == 'prep':
            edges.append(edge('cup','ON','table'))
            edges.append(edge('plate','ON','table'))
            edges.append(edge('pan','ON','counter'))
            edges.append(edge('coffee','ON','counter'))
            edges.append(edge('cheese','ON','table'))
            edges.append(edge('bread','ON','table'))
            edges.append(edge('eggs','ON','counter'))

        if time == 'after':
            edges.append(edge('cup','INSIDE','sink'))
            edges.append(edge('plate','INSIDE','sink'))
            edges.append(edge('pan','INSIDE','sink'))
            edges.append(edge('coffee','INSIDE','cabinet'))
            edges.append(edge('cheese','INSIDE','fridge'))
            edges.append(edge('bread','INSIDE','fridge'))
            edges.append(edge('eggs','INSIDE','fridge'))

    elif type == 'cereal':
        edges.append(edge('plate','INSIDE','cabinet'))
        edges.append(edge('pan','INSIDE','cabinet'))

        edges.append(edge('butter','INSIDE','fridge'))
        edges.append(edge('cheese','INSIDE','fridge'))
        edges.append(edge('bread','INSIDE','fridge'))
        edges.append(edge('eggs','INSIDE','fridge'))
        edges.append(edge('apple','ON','counter'))
        edges.append(edge('chocolate','ON','counter'))
        
        if time == 'prep':
            edges.append(edge('cup','ON','table'))
            edges.append(edge('bowl','ON','table'))
            edges.append(edge('spoon','INSIDE','bowl'))
            edges.append(edge('coffee','ON','counter'))
            edges.append(edge('cereal','ON','table'))
            edges.append(edge('milk','ON','table'))

        if time == 'after':
            edges.append(edge('cup','INSIDE','sink'))
            edges.append(edge('bowl','INSIDE','sink'))
            edges.append(edge('spoon','INSIDE','sink'))
            edges.append(edge('coffee','INSIDE','cabinet'))
            edges.append(edge('cereal','INSIDE','cabinet'))
            edges.append(edge('milk','INSIDE','fridge'))

    elif type == 'bread':
        edges.append(edge('bowl','INSIDE','cabinet'))
        edges.append(edge('spoon','INSIDE','cabinet'))
        edges.append(edge('pan','INSIDE','cabinet'))

        edges.append(edge('cereal','INSIDE','cabinet'))
        edges.append(edge('milk','INSIDE','fridge'))
        edges.append(edge('eggs','INSIDE','fridge'))
        edges.append(edge('apple','ON','counter'))
        edges.append(edge('chocolate','ON','counter'))
        
        if time == 'prep':
            edges.append(edge('cup','ON','table'))
            edges.append(edge('plate','ON','table'))
            edges.append(edge('coffee','ON','counter'))
            edges.append(edge('butter','ON','table'))
            edges.append(edge('cheese','ON','table'))
            edges.append(edge('bread','ON','table'))
        
        if time == 'after':
            edges.append(edge('cup','INSIDE','sink'))
            edges.append(edge('plate','INSIDE','sink'))
            edges.append(edge('coffee','INSIDE','cabinet'))
            edges.append(edge('butter','INSIDE','fridge'))
            edges.append(edge('cheese','INSIDE','fridge'))
            edges.append(edge('bread','INSIDE','fridge'))
    
    return edges


def time_external(in_t, dt=10):
    in_t = in_t*dt
    mins = in_t % 60
    in_t = in_t // 60
    hrs = in_t % 24
    in_t = in_t // 24
    days = in_t % 7
    in_t = in_t // 7
    weeks = in_t
    return(weeks, days, hrs, mins)

common_cfg = {
        'EPOCHS': 1000,
        'USE_SPECTRAL_LOSS': False
    }

def get_graph_sequence(type_idx = 1):
    nodes = [node_dictionary[key] for key in node_dictionary.keys()]
    return [{"nodes":nodes, "edges":get_edges(time_idx=t)} for t in range(4)]


def case1():
    """
    Same breakfast, stochastic time. 0.25 probability of wrong graph prediction
    """

    data_dir = 'data/mdptests/case1'
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)

    dt = 5
    num_routines = 100
    wrong_graph_prob = 0.25
    graph_sequence = get_graph_sequence()

    with open(os.path.join(data_dir,'classes.json'),'w') as f:
        json.dump({"nodes":graph_sequence[0]['nodes'], "edges": ["INSIDE", "ON"], "node_states": [["CLOSED","OPEN"],["OFF","ON"],["DIRTY","CLEAN"],["PLUGGED_OUT","PLUGGED_IN"]]}, f)

    with open(os.path.join(data_dir,'info.json'),'w') as f:
        json.dump({"num_routines":num_routines, "num_nodes":20, "num_movable_obj":14, "num_changing_nodes":6, "wrong_graph_prob":wrong_graph_prob, "dt": dt, "search_object_ids":[]}, f)

    start_time = time_mins(0,8,0,0)
    end_time = time_mins(0,10,0,0)
    change_time = [time_mins(30,8,0,0), time_mins(0,9,0,0), time_mins(30,9,0,0)]

    data = []

    while len(data) < num_routines:
        t = start_time
        state = 0
        times = [start_time]
        while t < end_time and state < 3:
            t += 1
            r = random()
            if r < wrong_graph_prob or (r < (1-wrong_graph_prob) and t >= change_time[state]):
                state += 1
                times.append(t)
        if len(times) == 4:
            data.append({"times":times, "graphs":graph_sequence, "objects_in_use":[[]]*len(times)})

    with open(os.path.join(data_dir,'routines.json'),'w') as f:
        json.dump(data, f)

    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg.update({'NAME':'MDP_single_breakfast'})

    run(cfg=cfg, path=data_dir)


if __name__ == '__main__':
    case1()