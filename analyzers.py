import torch
import numpy as np

class LossAnalyzer():
    """
    Base class to analyze the loss for useful metrics
    """
    def __init__(self):
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
    def __init__(self):
        super().__init__()
    def __call__(self, loss_tensor, **kwargs):
        return loss_tensor.mean()
    def name(self):
        return "Mean Loss"

class EdgeTypeLoss(LossAnalyzer):
    def __init__(self, edge_class_names):
        self.labels = edge_class_names
    def __call__(self, loss_tensor, **kwargs):
        assert(len(self.labels) == loss_tensor.size()[-1])
        losses_by_type = {self.labels[i]: loss_tensor[:,:,:,i].mean() for i in range(len(self.labels))}
        return losses_by_type
    def name(self):
        return "Loss by Edge Type"

class ChangedEdgeLoss(LossAnalyzer):
    def __init__(self):
        pass
    def __call__(self, loss_tensor, **kwargs):
        loss_at_change = loss_tensor[np.where(kwargs["x_edges"]!=kwargs["y_edges"])]
        mean_loss_at_change = loss_at_change.mean()
        return mean_loss_at_change
    def name(self):
        return "Mean Loss On Changed Edges"

class ChangedEdgeWeightedLoss(LossAnalyzer):
    def __init__(self, weight_changed_edges = 0.5):
        self.weight_changed_edges = weight_changed_edges
    def __call__(self, loss_tensor, **kwargs):
        loss_at_change = loss_tensor[np.where(kwargs["x_edges"]!=kwargs["y_edges"])]
        mean_loss_at_change = loss_at_change.mean()
        mean_loss = loss_tensor.mean()
        return self.weight_changed_edges * mean_loss_at_change + (1 - self.weight_changed_edges) * mean_loss
    def name(self):
        return "Mean Loss Weighted On Changed Edges"

class StaticGraphLoss(LossAnalyzer):
    def __init__(self, static_nodes):
        self.static_nodes = static_nodes
    def __call__(self, loss_tensor, **kwargs):
        loss_static = loss_tensor[self.static_nodes,self.static_nodes].mean()
        return loss_static
    def name(self):
        return "Mean Loss On Static Edges"