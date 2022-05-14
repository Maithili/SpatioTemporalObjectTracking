import random
from copy import deepcopy
import numpy as np
import torch
from torch.nn import functional as F


class Baseline():
    def __init__(self):
        pass

    def extract(self, batch):
        self.dynamic_edges = batch['dynamic_edges_mask']
        self.evaluate_node = self.dynamic_edges.sum(-1) > 0
        self.static_nodes_mask = (torch.unsqueeze(self.dynamic_edges.sum(-1) == 0, dim=-1)).expand(-1,-1,self.evaluate_node.size(1))
        self.edges = batch['edges']
        self.nodes = batch['nodes']
        self.context = batch['context']
        self.gt = batch['y_edges']
        self.time = batch['time']
    
    def step(self, batch):
        self.extract(batch)
        result = deepcopy(self.run())
        result[self.dynamic_edges == 0] = self.edges[self.dynamic_edges == 0]
        result = result/(result.sum(dim=-1).unsqueeze(-1).repeat(1,1,self.edges.size()[-1])+1e-8)
        assert result.squeeze(-1).argmax(-1).size() == self.edges.squeeze(-1).argmax(-1).size(), f"{result.size()} == {self.edges.size()}"
        details = {'input':{'class':self.nodes.argmax(-1), 'location':self.edges.squeeze(-1).argmax(-1)}, 
            'output_probs':{'location': result}, 
            'gt':{'class':self.nodes.argmax(-1), 'location': self.gt.squeeze(-1).argmax(-1)}, 
            'losses':{'location': None}, 
            'output':{'class':self.nodes.argmax(-1), 'location': result.squeeze(-1).argmax(-1)}, 
            'evaluate_node':self.evaluate_node}
        return None, details



class StateTimeConditionedBaseline(Baseline):
    def __init__(self):
        super().__init__()
    # def log(self):
    #     combined_cm_metrics = {k:sum([e['CM'][k] for e in self.eval]) for k in self.eval[0]['CM'].keys()}
    #     print(combined_cm_metrics)
    #     return {'Test accuracy':sum([e['accuracy'] for e in self.eval])/len(self.eval), 'Test CM metrics':combined_cm_metrics}

class TimeConditionedBaseline(Baseline):
    def __init__(self):
        super().__init__()
    # def log(self):
    #     return {'Test accuracy (Unconditional)':sum([e['accuracy'] for e in self.eval])/len(self.eval)}


class LastSeen(StateTimeConditionedBaseline):
    def __init__(self, cooccurence_freq) -> None:
        super().__init__()

    def run(self):
        return self.edges

class StaticSemantic(TimeConditionedBaseline):
    def __init__(self, cooccurence_freq) -> None:
        super().__init__()
        self.cooccurence_freq = cooccurence_freq

    def run(self):
        return self.cooccurence_freq.unsqueeze(0).repeat(self.edges.size()[0],1,1)

class LastSeenAndStaticSemantic(StateTimeConditionedBaseline):
    def __init__(self, cooccurence_freq, prob_change = 0.4) -> None:
        super().__init__()
        self.cooccurence_freq = cooccurence_freq
        self.prob_change = prob_change

    def run(self):
        next_edges = self.edges * (1-self.prob_change)
        next_edges += self.prob_change/next_edges.size()[-1]
        return self.cooccurence_freq * next_edges


class Fremen(TimeConditionedBaseline):
    def __init__(self, spectral_components):
        super().__init__()
        self.spectral_components = spectral_components
    
    def run(self):
        prior = sum([2*spec['amplitude']*np.cos(2*np.pi*self.time/spec['period'] - spec['phase']) for spec in self.spectral_components])
        return prior.unsqueeze(0).repeat(self.edges.size()[0],1,1)

class FremenStateConditioned(StateTimeConditionedBaseline):
    def __init__(self, spectral_components, dt, time_decay=30):
        super().__init__()
        self.spectral_components = spectral_components
        self.decay_exponent = np.exp(-dt / time_decay)
    
    def run(self):
        prior = sum([2*spec['amplitude']*np.cos(2*np.pi*self.time/spec['period'] - spec['phase']) for spec in self.spectral_components])
        posterior = prior + (self.edges-prior) * self.decay_exponent
        return posterior


class Slim(StateTimeConditionedBaseline):
    def __init__(self, cooccurence_freq, num_particles=10, noise=0.4) -> None:
        super().__init__()
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

    def run(self):
        particles = []
        weights = []

        for _ in range(self.num_particles):
            particles.append(self.add_noise(self.sample_scene(self.edges)))
            weights.append(torch.prod(((particles[-1]*self.cooccurence_freq).sum(-1)),0))

        return self.aggregate_particles(particles, weights)

