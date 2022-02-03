from lib2to3.pytree import Base
import random
import numpy as np
import torch

class Baseline():
    def __init__(self):
        pass
    def step(self, batch):
        pass
    def extract(self, batch):
        if 'dynamic_edges_mask' in batch.keys():
            dyn_edges = batch['dynamic_edges_mask']
            evaluate_node = dyn_edges.sum(-1) > 0
        self.evaluate_node = evaluate_node
        self.edges = batch['edges']
        self.context = batch['context']


class StateTimeConditionedBaseline(Baseline):
    def __init__(self):
        pass
    def step(self, batch):
        pass

class TimeConditionedBaseline(Baseline):
    def __init__(self):
        pass
    def step(self, batch):
        pass


class LastSeen(StateTimeConditionedBaseline):
    def __init__(self) -> None:
        pass

    def step(self, batch):
        self.extract(batch)
        edges = batch['edges']
        return edges

class StaticSemantic(TimeConditionedBaseline):
    def __init__(self, cooccurence_freq) -> None:
        self.cooccurence_freq = cooccurence_freq

    def step(self, _):
        return self.cooccurence_freq

class LastSeenAndStaticSemantic(StateTimeConditionedBaseline):
    def __init__(self, cooccurence_freq, prob_change = 0.5) -> None:
        self.cooccurence_freq = cooccurence_freq
        self.prob_change = prob_change

    def step(self, batch):
        self.extract(batch)
        next_edges = self.edges * (1-self.prob_change)
        next_edges += self.prob_change/next_edges.size()[-1]
        return self.cooccurence_freq * next_edges




class Slim(StateTimeConditionedBaseline):
    def __init__(self, cooccurence_freq, num_particles=10, noise=0.4) -> None:
        self.num_particles = num_particles
        self.noise = noise
        self.cooccurence_freq = cooccurence_freq

    def sample_scene(self, edges):
        prob_thresholds = torch.cumsum(edges, dim=-1)
        sample = random.random()
        _, idx = torch.max(prob_thresholds > sample, dim=-1)
        return torch.nn.functional.one_hot(idx, num_classes=edges.size()[-1])

    def add_noise(self, edges):
        new_edges = edges
        for i in range(new_edges.size()[0]):
            random_parents = torch.randint(0, edges.size()[-1], (edges.size()[-1],1))
            random_edges = torch.nn.functional.one_hot(random_parents, num_classes=edges.size()[-1])
            for j in range(new_edges.size()[1]):
                if random.random() < self.noise:
                    new_edges[i,j,:] = random_edges[j,:]
        return new_edges

    def aggregate_particles(self, particles, weights):
        weights = [w/sum(weights) for w in weights]
        mean_aggregate = torch.concat([particle*weight for particle, weight in zip(particles, weights)], dim=0).sum(dim=0)
        # mean_aggregate = mean_aggregate/mean_aggregate.sum(-1)
        sums = mean_aggregate.sum(-1).unsqueeze(-1)
        mean_aggregate = mean_aggregate/sums
        assert np.allclose(mean_aggregate.sum(-1), torch.ones_like(mean_aggregate.sum(-1))), mean_aggregate.sum(-1)
        return mean_aggregate

    def step(self, batch):
        self.extract(batch)
        particles = []
        weights = []

        for _ in range(self.num_particles):
            particles.append(self.add_noise(self.sample_scene(self.edges)))
            weights.append(torch.prod(((particles[-1]*self.cooccurence_freq).sum(-1)),0))

        return self.aggregate_particles(particles, weights)

