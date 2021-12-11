from logging import log
import torch
import numpy as np

class OutputFilters():
    def __init__(self, data, train_filter_nodes="mean", train_filter_edges="mean", log_filters_nodes=[], log_filters_edges=[], train_weight_nodes = 0.5):
        self.static_info = {}
        self.static_info["edge_classes"] = data.edge_keys
        self.static_info["weight_for_changed_edges"] = 0.9
        self.static_info["static_node_ids"] = data.static_nodes
        self.static_info["edges_of_interest"] = data.get_edges_of_interest()
        self.losses = {}
        self.filter_options_nodes = {
            "Mean": self.mean
        }
        self.filter_options_edges = {
            "Mean": self.mean,
            "Changed": self.changed_edges_mean,
            "Dynamic": self.dynamic_edges_mean,
            "Selected": self.selected_edge_losses,
        }
        self.train_filter = {'nodes':self.filter_options_nodes[train_filter_nodes], 
                             'edges':self.filter_options_edges[train_filter_edges]}
        self.log_filters = {'nodes':[(f,self.filter_options_nodes[f]) for f in log_filters_nodes],
                            'edges':[(f,self.filter_options_edges[f]) for f in log_filters_edges]}
        self.data_info = {}
        self.node_weight = train_weight_nodes
    
    def set_data_info(self, **kwargs):
        self.data_info = kwargs

    def train_metric(self, edges_tensor, nodes_tensor):
        node_loss = self.train_filter['nodes'](nodes_tensor)
        edge_loss = self.train_filter['edges'](edges_tensor)
        return self.node_weight * node_loss + (1 - self.node_weight) * edge_loss

    def logging_metrics(self, edges_tensor, nodes_tensor, prefix = ''):
        metrics = {prefix+"Nodes "+f[0]: f[1](nodes_tensor) for f in self.log_filters['nodes']}
        metrics.update({prefix+"Edges "+f[0]: f[1](edges_tensor) for f in self.log_filters['edges']})
        return metrics

    def mean(self, data):
        """
        Args : loss_tensor = batch_size x n_nodes ((x n_nodes)) x n_edge_types
        Returns : 1x1 statistic of loss
        """
        return data.mean()

    def changed_edges_mean(self, data):
        changes = (self.data_info["x_edges"]!=self.data_info["y_edges"]).sum(-1).unsqueeze(-1)>0
        mean_loss_at_change = data[changes].mean()
        return mean_loss_at_change

    def dynamic_edges_mean(self, data):
        nodes = self.data_info["node_classes"]
        nodes_in_graphs = (nodes).argmax(axis=-1)
        dyn_idx = np.logical_not(np.isin(nodes_in_graphs, self.static_info["static_node_ids"]))
        loss_dyn_idx = np.fromfunction(lambda b,i,j :  np.logical_or(dyn_idx[b,i], dyn_idx[b,j]), shape=nodes.shape, dtype=int)
        loss_dyn_idx = np.stack([loss_dyn_idx]*data.shape[-1],axis=-1)
        dyn_loss = (data[torch.from_numpy(loss_dyn_idx)]).mean()
        return dyn_loss

    def selected_edge_losses(self, data):
        nodes_in_graphs = (self.data_info["node_classes"]).argmax(axis=-1)
        loss_results = {}
        for n,i in self.idxs.items():
            graph0, idx0 = np.argwhere(nodes_in_graphs == i[0])
            graph1, idx1 = np.argwhere(nodes_in_graphs == i[1])
            assert all(graph0==graph1), "Do NOT use SpecificEdgeLoss when the nodes do not exist in every graph!"
            loss_results[n] = (data[graph0, idx0, idx1,i[2]]).mean()
        return loss_results
