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

def get_binary_graph(num_nodes):
    nodes = torch.Tensor(np.eye(num_nodes))
    edges = torch.zeros_like(nodes)
    for n in range(num_nodes):
        edges[n, ceil(n/2 - 1)] = 1
    return nodes, edges

class DataSplitDummy():
    def __init__(self, num_nodes = 10, dataset_size = 15):
        self.num_nodes = num_nodes
        self.collate_fn = CollateToDict(['edges', 'nodes', 'context', 'y_edges', 'y_nodes', 'dynamic_edges_mask', 'time', 'change_type'])
        self.contexts = [random.choice(np.arange(num_nodes-1))+1 for _ in range(dataset_size)]

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx: int):
        nodes, edges = get_binary_graph(self.num_nodes)
        y_edges = edges
        context = self.contexts[idx]
        y_edges[context,:] = 0
        y_edges[context, context-1] = 1
        dynamic_edges_mask = torch.ones_like(edges)
        dynamic_edges_mask[0,:] = 0
        return edges, nodes, torch.Tensor([context]), y_edges, nodes, dynamic_edges_mask, torch.Tensor(), torch.Tensor()

class DummyDataset():
    def __init__(self):
        n_nodes = 10
        self.train = DataSplitDummy(num_nodes=n_nodes, dataset_size=15)
        self.test = DataSplitDummy(num_nodes=n_nodes, dataset_size=15)
        self.params = {}
        self.params['n_nodes'] = n_nodes
        self.params['n_len'] = n_nodes
        self.params['c_len'] = 1

    def get_train_loader(self):
        return DataLoader(self.train, num_workers=8, batch_size=1, collate_fn=self.train.collate_fn)

    def get_test_loader(self):
        return DataLoader(self.test, num_workers=8, batch_size=1, collate_fn=self.test.collate_fn)

