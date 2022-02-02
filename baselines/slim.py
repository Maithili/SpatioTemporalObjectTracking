import random
import numpy as np
import torch


def sample_scene(edges):
    prob_thresholds = torch.cumsum(edges, dim=-1)
    sample = random.random()
    _, idx = torch.max(prob_thresholds > sample, dim=-1)
    return torch.nn.functional.one_hot(idx, num_classes=edges.size()[-1])

def add_noise(edges):
    new_edges = edges
    for i in range(new_edges.size()[0]):
        random_parents = torch.randint(0, edges.size()[-1], (edges.size()[-1],1))
        random_edges = torch.nn.functional.one_hot(random_parents, num_classes=edges.size()[-1])
        for j in range(new_edges.size()[1]):
            if random.random() < 0.1:
                new_edges[i,j,:] = random_edges[j,:]
    return new_edges

def aggregate_particles(particles, weights):
    weights = [w/sum(weights) for w in weights]
    mean_aggregate = torch.concat([particle*weight for particle, weight in zip(particles, weights)], dim=0).sum(dim=0)
    # mean_aggregate = mean_aggregate/mean_aggregate.sum(-1)
    sums = mean_aggregate.sum(-1).unsqueeze(-1)
    mean_aggregate = mean_aggregate/sums
    assert np.allclose(mean_aggregate.sum(-1), torch.ones_like(mean_aggregate.sum(-1))), mean_aggregate.sum(-1)
    return mean_aggregate

class Slim():
    def __init__(self, cooccurence_freq, num_particles=10) -> None:
        self.num_particles = num_particles
        self.cooccurence_freq = cooccurence_freq

    def step(self, batch):
        edges = batch['edges']
        # if 'dynamic_edges_mask' in batch.keys():
        #     dyn_edges = batch['dynamic_edges_mask']
        #     evaluate_node = dyn_edges.sum(-1) > 0
        #     assert np.isclose(edges.sum(dim=1)[evaluate_node], 1)

        particles = []
        weights = []

        for _ in range(self.num_particles):
            particles.append(add_noise(sample_scene(edges)))
            weights.append(torch.prod(((particles[-1]*self.cooccurence_freq).sum(-1)),0))

        return aggregate_particles(particles, weights)