import json
import numpy as np
from os import path as osp
import torch

def encode_time(t, dt):
    min_omega = 2*np.pi*dt/10
    omegas = [min_omega/(2**i) for i in range(5)]
    enc = []
    for om in omegas:
        enc = enc + [np.sin(om*t), np.cos(om*t)]
    return enc


def get_edge_data(graph, n_id1, n_id2, edge_classes):
    types = [0]*len(edge_classes)
    edges = [e for e in graph['edges'] if e['from_id']==n_id1 and e['to_id']==n_id2]
    for e in edges:
        types[edge_classes.index(e['relation_type'])] = 1
    return np.array(types)


def read_routine(node_classes, edge_classes, graphs, times):
    nodes = graphs[0]['nodes']
    node_ids = [n['id'] for n in nodes]
    node_features = np.zeros((len(nodes), len(node_classes)))
    for i,nid in enumerate(node_ids):
        node_features[i,:] = node_classes == nid

    edge_features = np.zeros((len(graphs), len(node_ids), len(node_ids), len(edge_classes)))
    adjacency = np.zeros((len(graphs), len(node_ids), len(node_ids)))
    for i,graph in enumerate(graphs):
        for j,n1 in enumerate(node_ids):
            for k,n2 in enumerate(node_ids):
                edge_features[i,j,k,:]= get_edge_data(graph, n1, n2, edge_classes)

    context = [encode_time(t, 10) for t in times]

    return torch.Tensor(node_features), torch.Tensor(edge_features), torch.Tensor(context)


def pairwise_data(nodes, edges, contexts):
    n = len(edges)

    data = []

    for i in range(n):
        for j in range(i+1,n):
            data.append((edges[i], nodes, contexts[i], contexts[j], edges[j]))
    return data

def read_data(data_path: str = 'data/example/data.json', classes_path='data/example/classes.json'):
    with open(classes_path, 'r') as f:
        classes = json.load(f)
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    node_classes = classes['nodes']
    edge_classes = classes['edges']

    training_data = []
    for routine in data:
        nodes, edges, contexts = read_routine(node_classes, edge_classes, routine["graphs"], routine["times"])
        training_data += pairwise_data(nodes, edges, contexts)

    return training_data
    