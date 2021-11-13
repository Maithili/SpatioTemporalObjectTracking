#!/home/maithili/repos/GraphTrans/.venv/bin/python

import os
import json
import shutil
from run import run

node_dictionary = {}
node_dictionary['kitchen'] = {"id": 1, "class_name": "kitchen", "category": "Rooms", "properties": [], "states": [], "prefab_name": None, "bounding_box": None}
node_dictionary['cabinet'] = {"id": 2, "class_name": "cabinet", "category": "Furniture", "properties": [], "states": ["CLOSED"], "prefab_name": None, "bounding_box": None}
node_dictionary['table'] = {"id": 3, "class_name": "table", "category": "Furniture", "properties": [], "states": ["CLOSED"], "prefab_name": None, "bounding_box": None}
node_dictionary['sink'] = {"id": 4, "class_name": "sink", "category": "Furniture", "properties": [], "states": ["CLOSED"], "prefab_name": None, "bounding_box": None}
node_dictionary['cup'] = {"id": 5, "class_name": "cup", "category": "placable_objects", "properties": [], "states": ["CLEAN"], "prefab_name": None, "bounding_box": None}
node_dictionary['cereal'] = {"id": 6, "class_name": "cereal", "category": "placable_objects", "properties": [], "states": ["CLEAN"], "prefab_name": None, "bounding_box": None}
node_dictionary['toast'] = {"id": 7, "class_name": "toast", "category": "placable_objects", "properties": [], "states": ["CLEAN"], "prefab_name": None, "bounding_box": None}
node_dictionary['bread'] = {"id": 8, "class_name": "bread", "category": "placable_objects", "properties": [], "states": ["CLEAN"], "prefab_name": None, "bounding_box": None}

def edge(from_node, relation, to_node):
    return {'from_id':node_dictionary[from_node]['id'], 'relation_type':relation, 'to_id':node_dictionary[to_node]['id']}

def time_internal(mins, hrs, days=0, weeks=0, dt=10):
    return int(round(((((weeks*7+days)*24)+hrs)*60+mins)/dt))

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
        'SUFFIX': " only Accuracy Loss",
        'EPOCHS': 250,
        'SEQUENTIAL_PREDICTION': True,
        'USE_SPECTRAL_LOSS': False
    }

def case1():
    """
    No noise repeating case
    """

    data_dir = 'data/unittests/case1'
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)

    nodes = [node_dictionary[key] for key in ['kitchen','cabinet','table','sink','cup']]

    with open(os.path.join(data_dir,'classes.json'),'w') as f:
        json.dump({"nodes":nodes, "edges": ["INSIDE"]}, f)

    data = []

    fixed_edges = []
    fixed_edges.append(edge('cabinet','INSIDE','kitchen'))
    fixed_edges.append(edge('table','INSIDE','kitchen'))
    fixed_edges.append(edge('sink','INSIDE','kitchen'))

    edges_0 = []
    edges_0.append(edge('cup','INSIDE','cabinet'))

    edges_1 = []
    edges_1.append(edge('cup','INSIDE','table'))

    edges_2 = []
    edges_2.append(edge('cup','INSIDE','sink'))

    changing_edges = [edges_0, edges_1, edges_2, edges_0]

    graph_times = [time_internal(0,8,0,0), time_internal(20,8,0,0), time_internal(40,8,0,0), time_internal(0,9,0,0)]
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges]})
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges]})
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges]})
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges]})
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges]})

    with open(os.path.join(data_dir,'sample.json'),'w') as f:
        json.dump(data, f)

    cfg = common_cfg
    cfg.update({
        'DATA_PATH': os.path.join(data_dir,'sample.json'),
        'CLASSES_PATH': os.path.join(data_dir,'classes.json'),
        'NAME': "Test - Single edge type"+common_cfg['SUFFIX'],
        'EDGES_OF_INTEREST': [('cup','INSIDE','cabinet'), 
                              ('cup','INSIDE','table'), 
                              ('cup','INSIDE','sink')],
        'TIME_START': [7,50],
        'TIME_END': [9,10],
    })
    run(cfg)



def case2():
    """
    No noise repeating case with multiple edge types
    """

    data_dir = 'data/unittests/case2'
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)
    
    nodes = [node_dictionary[key] for key in ['kitchen','cabinet','table','sink','cup']]

    with open(os.path.join(data_dir,'classes.json'),'w') as f:
        json.dump({"nodes":nodes, "edges": ["CLOSE","ON","INSIDE"]}, f)

    data = []

    fixed_edges = []
    fixed_edges.append(edge('cabinet','INSIDE','kitchen'))
    fixed_edges.append(edge('table','INSIDE','kitchen'))
    fixed_edges.append(edge('sink','INSIDE','kitchen'))

    edges_0 = []
    edges_0.append(edge('cup','INSIDE','cabinet'))
    edges_0.append(edge('cup','CLOSE','cabinet'))
    edges_0.append(edge('cabinet','CLOSE','cup'))

    edges_1 = []
    edges_1.append(edge('cup','ON','table'))
    edges_1.append(edge('cup','CLOSE','table'))
    edges_1.append(edge('table','CLOSE','cup'))

    edges_2 = []
    edges_2.append(edge('cup','INSIDE','sink'))
    edges_2.append(edge('cup','CLOSE','sink'))
    edges_2.append(edge('sink','CLOSE','cup'))

    changing_edges = [edges_0, edges_1, edges_2, edges_0]

    graph_times = [time_internal(0,8,0,0), time_internal(20,8,0,0), time_internal(40,8,0,0), time_internal(0,9,0,0)]
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges]})
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges]})
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges]})
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges]})
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges]})

    with open(os.path.join(data_dir,'sample.json'),'w') as f:
        json.dump(data, f)

    cfg = common_cfg
    cfg.update({
        'DATA_PATH': os.path.join(data_dir,'sample.json'),
        'CLASSES_PATH': os.path.join(data_dir,'classes.json'),
        'NAME': "Test - No noise"+common_cfg['SUFFIX'],
        'EDGES_OF_INTEREST': [('cup','INSIDE','cabinet'), 
                              ('cup','ON','table'), 
                              ('cup','INSIDE','sink')],
        'TIME_START': [7,50],
        'TIME_END': [9,10]
    })
    run(cfg)

    
def case3():
    """
    No noise repeating case with different times
    """

    data_dir = 'data/unittests/case3'
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)

    nodes = [node_dictionary[key] for key in ['kitchen','cabinet','table','sink','cup']]

    with open(os.path.join(data_dir,'classes.json'),'w') as f:
        json.dump({"nodes":nodes, "edges": ["INSIDE"]}, f)

    data = []

    fixed_edges = []
    fixed_edges.append(edge('cabinet','INSIDE','kitchen'))
    fixed_edges.append(edge('table','INSIDE','kitchen'))
    fixed_edges.append(edge('sink','INSIDE','kitchen'))

    edges_0 = []
    edges_0.append(edge('cup','INSIDE','cabinet'))

    edges_1 = []
    edges_1.append(edge('cup','INSIDE','table'))

    edges_2 = []
    edges_2.append(edge('cup','INSIDE','sink'))

    changing_edges = [edges_0, edges_1, edges_2, edges_0]

    graph_times = [time_internal(0,7,0,0), time_internal(20,7,0,0), time_internal(40,7,0,0), time_internal(0,8,0,0)]
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges]})
    graph_times = [time_internal(10,7,0,0), time_internal(30,7,0,0), time_internal(50,7,0,0), time_internal(10,8,0,0)]
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges]})
    graph_times = [time_internal(0,8,0,0), time_internal(20,8,0,0), time_internal(40,8,0,0), time_internal(0,9,0,0)]
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges]})
    graph_times = [time_internal(10,8,0,0), time_internal(30,8,0,0), time_internal(50,8,0,0), time_internal(10,9,0,0)]
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges]})
    graph_times = [time_internal(30,8,0,0), time_internal(50,8,0,0), time_internal(10,9,0,0), time_internal(30,9,0,0)]
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges]})

    with open(os.path.join(data_dir,'sample.json'),'w') as f:
        json.dump(data, f)

    cfg = common_cfg
    cfg.update({
        'DATA_PATH': os.path.join(data_dir,'sample.json'),
        'CLASSES_PATH': os.path.join(data_dir,'classes.json'),
        'NAME': "Test - Changing context"+common_cfg['SUFFIX'],
        'EDGES_OF_INTEREST': [('cup','INSIDE','cabinet'), 
                              ('cup','INSIDE','table'), 
                              ('cup','INSIDE','sink')],
        'EPOCHS': 500,
        'TIME_START': [6,50],
        'TIME_END': [9,40]
    })
    run(cfg)

def case4():
    """
    50-50 repeating case
    """

    data_dir = 'data/unittests/case4'
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)


    nodes = [node_dictionary[key] for key in ['kitchen','cabinet','table','cereal','toast']]

    with open(os.path.join(data_dir,'classes.json'),'w') as f:
        json.dump({"nodes":nodes, "edges": ["ON","INSIDE","CLOSE"]}, f)

    data = []

    fixed_edges = []
    fixed_edges.append(edge('cabinet','INSIDE','kitchen'))
    fixed_edges.append(edge('table','INSIDE','kitchen'))
    fixed_edges.append(edge('sink','INSIDE','kitchen'))

    edges_0 = []
    edges_0.append(edge('cereal','INSIDE','cabinet'))
    edges_0.append(edge('cereal','CLOSE','cabinet'))
    edges_0.append(edge('toast','INSIDE','cabinet'))
    edges_0.append(edge('toast','CLOSE','cabinet'))

    edges_1a = []
    edges_1a.append(edge('toast','INSIDE','cabinet'))
    edges_1a.append(edge('toast','CLOSE','cabinet'))
    edges_1a.append(edge('cereal','ON','table'))
    edges_1a.append(edge('cereal','CLOSE','table'))

    edges_1b = []
    edges_1b.append(edge('cereal','INSIDE','cabinet'))
    edges_1b.append(edge('cereal','CLOSE','cabinet'))
    edges_1b.append(edge('toast','ON','table'))
    edges_1b.append(edge('toast','CLOSE','table'))


    changing_edges_a = [edges_0, edges_1a]
    changing_edges_b = [edges_0, edges_1b]
    dt = 20

    graph_times = [time_internal(0,8,0,0, dt=dt), time_internal(20,8,0,0, dt=dt)]
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges_a]})
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges_b]})
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges_a]})
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges_b]})
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges_a]})
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges_b]})
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges_a]})
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges_b]})
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges_a]})
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges_b]})

    with open(os.path.join(data_dir,'sample.json'),'w') as f:
        json.dump(data, f)

    cfg = common_cfg
    cfg.update({
        'DATA_PATH': os.path.join(data_dir,'sample.json'),
        'CLASSES_PATH': os.path.join(data_dir,'classes.json'),
        'NAME': "Test - 50/50"+common_cfg['SUFFIX'],
        'DT': dt,
        'EDGES_OF_INTEREST': [('cereal','INSIDE','cabinet'), 
                              ('cereal','ON','table'), 
                              ('toast','INSIDE','cabinet'), 
                              ('toast','ON','table')],
        'TIME_START': [8,00],
        'TIME_END': [8,20],
    })
    run(cfg)

def case5():
    """
    Learning from context ONLY
    """

    data_dir = 'data/unittests/case5'
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)


    nodes = [node_dictionary[key] for key in ['kitchen','cabinet','table','cereal','toast','bread']]
    print(nodes)

    with open(os.path.join(data_dir,'classes.json'),'w') as f:
        json.dump({"nodes":nodes, "edges": ["ON","INSIDE","CLOSE"]}, f)

    data = []

    fixed_edges = []
    fixed_edges.append(edge('cabinet','INSIDE','kitchen'))
    fixed_edges.append(edge('table','INSIDE','kitchen'))

    edges_0 = []
    edges_0.append(edge('cereal','INSIDE','cabinet'))
    edges_0.append(edge('cereal','CLOSE','cabinet'))
    edges_0.append(edge('toast','INSIDE','cabinet'))
    edges_0.append(edge('toast','CLOSE','cabinet'))
    edges_0.append(edge('bread','INSIDE','cabinet'))
    edges_0.append(edge('bread','CLOSE','cabinet'))

    edges_1a = []
    edges_1a.append(edge('toast','INSIDE','cabinet'))
    edges_1a.append(edge('toast','CLOSE','cabinet'))
    edges_1a.append(edge('cereal','ON','table'))
    edges_1a.append(edge('cereal','CLOSE','table'))
    edges_1a.append(edge('bread','INSIDE','cabinet'))
    edges_1a.append(edge('bread','CLOSE','cabinet'))

    edges_1b = []
    edges_1b.append(edge('cereal','INSIDE','cabinet'))
    edges_1b.append(edge('cereal','CLOSE','cabinet'))
    edges_1b.append(edge('toast','ON','table'))
    edges_1b.append(edge('toast','CLOSE','table'))
    edges_1b.append(edge('bread','INSIDE','cabinet'))
    edges_1b.append(edge('bread','CLOSE','cabinet'))

    edges_1c = []
    edges_1c.append(edge('cereal','INSIDE','cabinet'))
    edges_1c.append(edge('cereal','CLOSE','cabinet'))
    edges_1c.append(edge('toast','ON','cabinet'))
    edges_1c.append(edge('toast','CLOSE','cabinet'))
    edges_1c.append(edge('bread','INSIDE','table'))
    edges_1c.append(edge('bread','CLOSE','table'))

    changing_edges_a = [edges_0, edges_1a]
    changing_edges_b = [edges_0, edges_1b]
    changing_edges_c = [edges_0, edges_1c]

    graph_times = [time_internal(0,8,0,0), time_internal(20,8,0,0)]
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges_a]})
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges_a]})
    graph_times = [time_internal(0,8,0,0), time_internal(30,8,0,0)]
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges_b]})
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges_b]})
    graph_times = [time_internal(0,8,0,0), time_internal(40,8,0,0)]
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges_c]})
    data.append({"times":graph_times, "graphs":[{"nodes":nodes, "edges":edges+fixed_edges} for edges in changing_edges_c]})

    with open(os.path.join(data_dir,'sample.json'),'w') as f:
        json.dump(data, f)

    cfg = common_cfg
    cfg.update({
        'DATA_PATH': os.path.join(data_dir,'sample.json'),
        'CLASSES_PATH': os.path.join(data_dir,'classes.json'),
        'NAME': "Test - Context Only"+common_cfg['SUFFIX'],
        'EDGES_OF_INTEREST': [('cereal','INSIDE','cabinet'), 
                              ('cereal','ON','table'), 
                              ('toast','INSIDE','cabinet'), 
                              ('toast','ON','table'), 
                              ('bread','INSIDE','cabinet'), 
                              ('bread','ON','table')],
        'TIME_START': [7,50],
        'TIME_END': [8,50],
    })
    run(cfg)


if __name__ == '__main__':
    case1()
    case2()
    case3()
    case4()
    case5()