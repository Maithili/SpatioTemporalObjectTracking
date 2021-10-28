import torch

class LossAnalyzer():
    """
    Base class to analyze the loss for useful metrics
    """
    def __init__(self):
        pass

    def __call__(self, loss_tensor):
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
    def __call__(self, loss_tensor):
        return loss_tensor.mean()
    def name(self):
        return "Mean Loss"

class EdgeTypeLoss(LossAnalyzer):
    def __init__(self, edge_class_names):
        self.labels = edge_class_names
    def __call__(self, loss_tensor):
        assert(len(self.labels) == loss_tensor.size()[-1])
        losses_by_type = {self.labels[i]: loss_tensor[:,:,:,i].mean() for i in range(len(self.labels))}
        return losses_by_type
    def name(self):
        return "Loss by Edge Type"

class StaticGraphLoss(LossAnalyzer):
    pass