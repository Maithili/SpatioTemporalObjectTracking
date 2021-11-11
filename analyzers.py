import torch
import numpy as np

class LossAnalyzer():
    """
    Base class to analyze the loss for useful metrics
    """
    def __init__(self, **kwargs):
        pass

    def __call__(self, loss_tensor, **kwargs):
        """
        Args : loss_tensor = batch_size x n_nodes x n_nodes x n_edge_types
        Returns : 1x1 statistic of loss
        """
        return torch.Tensor([0])

    def name(self):
        return "Zero"

class MeanLoss(LossAnalyzer):
    def __init__(self, **kwargs):
        super().__init__()
    def __call__(self, loss_tensor, **kwargs):
        return loss_tensor.mean()
    def name(self):
        return "Mean Loss"

class EdgeTypeLoss(LossAnalyzer):
    def __init__(self, **kwargs):
        self.labels = kwargs["edge_classes"]
    def __call__(self, loss_tensor, **kwargs):
        assert(len(self.labels) == loss_tensor.size()[-1])
        losses_by_type = {self.labels[i]: loss_tensor[:,:,:,i].mean() for i in range(len(self.labels))}
        return losses_by_type
    def name(self):
        return "Loss by Edge Type"

class MeanLossWhereExists(LossAnalyzer):
    def __init__(self, **kwargs):
        pass
    def __call__(self, loss_tensor, **kwargs):
        mean_loss_where_exists = loss_tensor[np.where(kwargs["y_edges"])].mean()
        return mean_loss_where_exists
    def name(self):
        return "Mean Loss Where Edge Exists"

class EdgeTypeLossWhereExists(LossAnalyzer):
    def __init__(self, **kwargs):
        self.labels = kwargs["edge_classes"]
    def __call__(self, loss_tensor, **kwargs):
        assert(len(self.labels) == loss_tensor.size()[-1])
        losses_by_type = {self.labels[i]: loss_tensor[:,:,:,i][kwargs["y_edges"][:,:,:,i]>0].mean() for i in range(len(self.labels))}
        return losses_by_type
    def name(self):
        return "Loss by Edge Type Where Edge Exists"

class ChangedEdgeLoss(LossAnalyzer):
    def __init__(self, **kwargs):
        pass
    def __call__(self, loss_tensor, **kwargs):
        loss_at_change = loss_tensor[np.where(kwargs["x_edges"]!=kwargs["y_edges"])]
        mean_loss_at_change = loss_at_change.mean()
        return mean_loss_at_change
    def name(self):
        return "Mean Loss On Changed Edges"

class ChangedEdgeWeightedLoss(LossAnalyzer):
    def __init__(self, **kwargs):
        self.weight_changed_edges = kwargs["weight_for_changed_edges"]
    def __call__(self, loss_tensor, **kwargs):
        loss_at_change = loss_tensor[np.where(kwargs["x_edges"]!=kwargs["y_edges"])]
        mean_loss_at_change = loss_at_change.mean()
        mean_loss = loss_tensor.mean()
        return self.weight_changed_edges * mean_loss_at_change + (1 - self.weight_changed_edges) * mean_loss
    def name(self):
        return "Mean Loss Weighted On Changed Edges"

class StaticGraphLoss(LossAnalyzer):
    def __init__(self, **kwargs):
        self.static_nodes = kwargs["static_nodes"]
    def __call__(self, loss_tensor, **kwargs):
        loss_static = loss_tensor[self.static_nodes,self.static_nodes].mean()
        return loss_static
    def name(self):
        return "Mean Loss On Static Edges"

class SpecificEdgeLoss(LossAnalyzer):
    """
    Use this only when the node is known to be in all graphs. Else behavior can be undefined!
    """
    def __init__(self, **kwargs):
        self.idxs = kwargs["edges_of_interest"]
        print(self.idxs)
    def __call__(self, loss_tensor, **kwargs):
        nodes_in_graphs = (kwargs["nodes"]).argmax(axis=-1)
        loss_results = {}
        for n,i in self.idxs.items():
            graph0, idx0 = np.argwhere(nodes_in_graphs == i[0])
            graph1, idx1 = np.argwhere(nodes_in_graphs == i[1])
            assert all(graph0==graph1), "Do NOT use SpecificEdgeLoss when the nodes do not exist in every graph!"
            loss_results[n] = (loss_tensor[graph0, idx0, idx1,i[2]]).mean()
        return loss_results
    def name(self):
        return "Specific Edge Loss"

class loss_options():
    def __init__(self, data):
        self.options = {x.__name__:x for x in LossAnalyzer.__subclasses__()}
        args = {}
        args["edge_classes"] = data.edge_keys
        args["weight_for_changed_edges"] = 0.9
        args["static_nodes"] = data.static_nodes
        args["edges_of_interest"] = data.get_edges_of_interest()
        self.losses = {}
        for name,clas in self.options.items():
            self.losses[name] = clas(**args)
    def __call__(self, name):
        return self.losses[name]
    def __contains__(self, name):
        return name in self.losses