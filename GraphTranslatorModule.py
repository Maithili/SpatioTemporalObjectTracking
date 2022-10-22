from importlib.util import module_for_loader
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import json
import sys
sys.path.append('helpers')
from random import random
import dill as pickle
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from pytorch_lightning.core.lightning import LightningModule
import torch_geometric.nn as geom_nn

from breakdown_evaluations import evaluate_all_breakdowns, activity_list


def get_masks(gt_tensor, output_tensor, input_tensor):
    masks = {}
    masks['gt_negatives'] = (gt_tensor == input_tensor).cpu()
    masks['gt_positives'] = (gt_tensor != input_tensor).cpu()
    masks['out_negatives'] = (output_tensor == input_tensor).cpu()
    masks['out_positives'] = (output_tensor != input_tensor).cpu()
    masks['tp'] = np.bitwise_and(masks['out_positives'], masks['gt_positives']).to(bool)
    masks['fp'] = np.bitwise_and(masks['out_positives'], masks['gt_negatives']).to(bool)
    masks['tn'] = np.bitwise_and(masks['out_negatives'], masks['gt_negatives']).to(bool)
    masks['fn'] = np.bitwise_and(masks['out_negatives'], masks['gt_positives']).to(bool)
    masks['correct'] = gt_tensor == output_tensor
    masks['wrong'] = gt_tensor != output_tensor
    return masks

# def evaluate_accuracy(gt_tensor, loss_tensor, output_tensor, input_tensor):
#     masks = get_masks(gt_tensor, output_tensor, input_tensor)
#     result = {}
#     result['accuracy'] = (masks['correct'].sum())/torch.numel(gt_tensor)
#     if loss_tensor is not None:
#         result['losses'] = {}
#         result['losses']['mean'] = loss_tensor.mean()
#         for key, mask in masks.items():
#             result['losses'][key] = loss_tensor[mask].sum()/mask.sum()
#     return result

# def evaluate(gt, output, input, evaluate_node, losses=None):
#     gt_tensor = gt[evaluate_node]
#     input_tensor = input[evaluate_node]
#     output_tensor = output[evaluate_node]
#     if losses is not None:
#         loss_tensor = losses[evaluate_node]
#         result = evaluate_accuracy(gt_tensor, loss_tensor, output_tensor, input_tensor)
#     else:
#         result = evaluate_accuracy(gt_tensor, None, output_tensor, input_tensor)
#     return result


class GraphTranslatorModule(LightningModule):
    def __init__(self, model_configs):
        
        super().__init__()

        self.cfg = model_configs
        self.n_nodes  = model_configs.n_nodes 
        self.n_len = model_configs.n_len
        self.c_len = model_configs.c_len
        self.edge_importance = model_configs.edge_importance
        self.context_dropout_probs = model_configs.context_dropout_probs ## change to context_dropout_probs
        self.learned_time_periods = model_configs.learned_time_periods
        self.preprocess_context = model_configs.preprocess_context

        self.hidden_influence_dim = 20

        self.edges_update_input_dim = self.hidden_influence_dim*5 + self.c_len
        self.nodes_update_input_dim = self.hidden_influence_dim*2 + self.n_len + self.c_len
        
        if model_configs.learned_time_periods:
            self.period_in_days = torch.nn.Parameter(torch.randn((1, int(self.c_len/2))))
            omega_one_day = torch.Tensor([2*np.pi/60*24])
            self.context_from_time = lambda t : torch.cat((torch.cos(omega_one_day * t / self.period_in_days),torch.sin(omega_one_day * t / self.period_in_days)), axis=1)
 
        self.mlp_context = nn.Sequential(nn.Linear(self.c_len, self.c_len),
                                             nn.ReLU(),
                                             nn.Linear(self.c_len, self.c_len),
                                             )


        mlp_hidden = model_configs.hidden_layer_size

        self.mlp_influence = nn.Sequential(nn.Linear(2*self.n_len+1, mlp_hidden),
                                                    nn.ReLU(),
                                                    nn.Linear(mlp_hidden, self.hidden_influence_dim),
                                                    )

        self.mlp_update_importance = nn.Sequential(nn.Linear(self.edges_update_input_dim, self.hidden_influence_dim),
                                                    nn.ReLU(),
                                                    nn.Linear(self.hidden_influence_dim, 1)
                                                    )
                                    
        self.mlp_update_edges = nn.Sequential(nn.Linear(self.edges_update_input_dim, self.hidden_influence_dim),
                                                    nn.ReLU(),
                                                    nn.Linear(self.hidden_influence_dim, 1)
                                                    )

    
        self.location_loss = lambda x,y: (nn.CrossEntropyLoss(reduction='none')(x.squeeze(-1).permute(0,2,1), y.squeeze(-1).long()))
        self.inference_location = lambda x: x.squeeze(-1).argmax(-1)

        self.class_loss = lambda xc,yc: nn.CrossEntropyLoss(reduction='none')(xc.permute(0,2,1), yc.long())
        self.inference_class = lambda xc: xc.argmax(-1)



    def inference_location_comparative(self, probs, ref, thresh=0.2):
        non_comp_inf = F.one_hot(self.inference_location(probs))
        ref_onehot = F.one_hot(ref)
        not_good_enough = ((probs[non_comp_inf] - probs[ref_onehot]) < thresh).sum(-1)
        non_comp_inf[not_good_enough] = ref[not_good_enough]
        return non_comp_inf

    def get_time_context(self, t, context_time):
        if self.learned_time_periods:
            input('Are you really trying to learn time periods?')
            time = t.unsqueeze(1)
            time_context = self.context_from_time(time)
        else:
            time_context = context_time
        
        time_context = self.mlp_context(time_context)

        return time_context.view(-1, self.c_len)

    def forward(self, edges, nodes, context):
        """
        Args:
            adjacency: batch_size x from_nodes x to_nodes x 1
            edge_features: batch_size x from_nodes x to_nodes x edge_feature_len
            nodes: batch_size x num_nodes x node_feature_len
            context_curr: batch_size x context_len
            context_query: batch_size x context_len
        """

        batch_size, num_nodes, node_feature_len = nodes.size()
        batch_size_e, num_f_nodes, num_t_nodes = edges.size()
        
        # Sanity check input dimensions
        assert batch_size == batch_size_e, "Different edge and node batch sizes"
        assert self.n_len == node_feature_len, (str(self.n_len) +'!='+ str(node_feature_len))
        assert self.n_nodes == num_nodes, (str(self.n_nodes) +'!='+ str(num_nodes))
        assert self.n_nodes == num_f_nodes, (str(self.n_nodes) +'!='+ str(num_f_nodes))
        assert self.n_nodes == num_t_nodes, (str(self.n_nodes) +'!='+ str(num_t_nodes))
        batch_size, num_nodes, node_feature_len = nodes.size()

        context = context.view(size=[batch_size, self.c_len])

        x = self.collate_edges(edges=edges.unsqueeze(-1), nodes=nodes)
        x = x.view(
            size=[batch_size * self.n_nodes * self.n_nodes, 
                  2*self.n_len+1])
        x = self.mlp_influence(x)
        x = x.view(
            size=[batch_size, 
                  self.n_nodes, 
                  self.n_nodes, 
                  self.hidden_influence_dim])

        if self.edge_importance == 'predicted':
            ## importance update
            imp = self.message_collection_edges(x, edges.unsqueeze(-1), context)
            imp = imp.view(
                size=[batch_size * self.n_nodes * self.n_nodes, 
                    self.edges_update_input_dim])
            imp = self.mlp_update_importance(imp).view(size=[batch_size, 
                                            self.n_nodes, 
                                            self.n_nodes,
                                            1])
        elif self.edge_importance == 'all':
            imp = torch.ones_like(edges.unsqueeze(-1))
        elif self.edge_importance == 'existing':
            imp = edges.unsqueeze(-1)
        else:
            raise KeyError(f'Edge Importance given as ({self.edge_importance}) is not among predicted, all or existing')

        ## edge message passing
        xe = self.message_collection_edges(x, imp, context)
        
        ## edge update
        xe = xe.view(
            size=[batch_size * self.n_nodes * self.n_nodes, 
                self.edges_update_input_dim])
        xe = self.mlp_update_edges(xe).view(size=[batch_size, 
                                                self.n_nodes, 
                                                self.n_nodes])


        # ## node update
        # xn = self.message_collection_nodes(x, imp, nodes, context)
        # xn = xn.view(
        #     size=[batch_size * self.num_nodes * self.num_nodes, 
        #         self.edges_update_input_dim])
        # xn = self.mlp_update_nodes(xn).view(size=[batch_size, 
        #                                 self.num_nodes, 
        #                                 self.num_nodes,
        #                                 1])
        
        imp = imp.view(size=[batch_size, self.n_nodes, self.n_nodes])

        # edges_inferred = F.softmax(xe, dim=-1)
        edges_inferred = xe

        return edges_inferred, nodes, imp


    def step(self, batch):
        edges = batch['edges']
        nodes = batch['nodes']
        y_edges = batch['y_edges']
        y_nodes = batch['y_nodes']
        dyn_edges = batch['dynamic_edges_mask']
        batch_size = nodes.size()[0]
        
        time_context = self.get_time_context(batch['time'], batch['context_time'])
        
        context = time_context
 
        edges_pred, nodes_pred, imp = self(edges, nodes, context)

        assert edges_pred.size() == dyn_edges.size(), f'Size mismatch in edges {edges_pred.size()} and dynamic mask {dyn_edges.size()}'
        edges_pred[dyn_edges == 0] = -float('inf')

        edges_inferred = F.softmax(edges_pred, dim=-1)
        edges_inferred[dyn_edges == 0] = edges[dyn_edges == 0]

        evaluate_node = dyn_edges.sum(-1) > 0

        input = {'class':self.inference_class(nodes), 
                 'location':self.inference_location(edges)}
                 
        output_probs = {'class':nodes_pred, 
                        'location':edges_inferred}

        gt = {'class':self.inference_class(y_nodes), 
              'location':self.inference_location(y_edges)}

        losses = {'class':self.class_loss(output_probs['class'], gt['class']),
                  'location':self.location_loss(edges_pred, gt['location']),
                  }

        output = {'class':self.inference_class(output_probs['class']),
                  'location':self.inference_location(edges_inferred)}
        

        # for result, name in zip([input, gt, losses, output], ['input', 'gt', 'losses', 'output']):
        #     assert list(result['class'].size()) == list(nodes.size())[:2], 'wrong class size for {} : {} vs {}'.format(name, result['class'].size(), nodes.size())
        #     assert list(result['location'].size()) == list(nodes.size())[:2], 'wrong class size for {} : {} vs {}'.format(name, result['class'].size(), nodes.size())

        # assert list(output_probs['class'].size())[:-1] == list(nodes.size())[:-1], 'wrong class size for probs : {} vs {}'.format(output_probs['class'].size(), nodes.size())
        # assert list(output_probs['location'].size()) == list(edges.size()), 'wrong class size for probs : {} vs {}'.format(output_probs['class'].size(), edges.size())

        # eval = evaluate(gt=gt['location'], output=output['location'], input=input['location'], evaluate_node=evaluate_node, losses=losses['location'])

        loss = {'graph': losses['location'][evaluate_node].mean(), 'context_process': losses['context_process'].mean(), 'context_predict': losses['context_predict'].mean()}

        assert (sum(loss.values()).size() == torch.Size([]))

        details = {'input':input, 
                   'output_probs':output_probs, 
                   'gt':gt, 
                   'losses':losses, 
                   'output':output, 
                   'evaluate_node':evaluate_node,
                   'importance_weights':imp,
                   }

        return loss, details

        
    def training_step(self, batch, batch_idx):
        loss, details = self.step(batch)
        self.log('Train loss',loss)
        self.log('Train',details['logs'])
        assert (sum(loss.values())).size() == torch.Size([])
        return sum(loss.values())

    def test_step(self, batch, batch_idx):
        loss, details = self.step(batch)
        self.log('Test loss',loss)
        self.log('Test',details['logs'])
        return 

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def collate_edges(self, edges, nodes):
        # nodes_repeated : batch_size x nodes x repeat dimension x node_feature_len
        nodes_repeated = nodes.unsqueeze(2).repeat([1,1,self.n_nodes,1])
        # concatenated : batch_size x from_nodes x to_nodes x (node_feature * 2 + edge_feature)
        concatenated = torch.cat([nodes_repeated, nodes_repeated.permute(0,2,1,3), edges], dim=-1)
        assert(len(concatenated.size())==4)
        assert(concatenated.size()[1]==self.n_nodes)
        assert(concatenated.size()[2]==self.n_nodes)
        assert(concatenated.size()[3]==self.n_len*2+1)
        return concatenated

    def message_collection_edges(self, edge_influence, edges, context):
        # context = batch_size x context_length
        # edge_influence : batch_size x from_nodes x to_nodes x hidden_influence_dim

        masked_edge_influence = torch.mul(edge_influence,edges)

        # batch_size x nodes x 1 x hidden_influence_dim
        from_from_influence = masked_edge_influence.sum(dim=2).unsqueeze(2).repeat([1,1,self.n_nodes,1])
        from_to_influence = masked_edge_influence.sum(dim=1).unsqueeze(2).repeat([1,1,self.n_nodes,1])
        # batch_size x 1 x nodes x hidden_influence_dim
        to_to_influence = masked_edge_influence.sum(dim=1).unsqueeze(1).repeat([1,self.n_nodes,1,1])
        to_from_influence = masked_edge_influence.sum(dim=2).unsqueeze(1).repeat([1,self.n_nodes,1,1])
        
        # all_influences : batch_size x from_nodes x to_nodes x hidden_influence_dim
        all_influences = torch.cat([from_from_influence, from_to_influence, to_to_influence, to_from_influence],dim=-1)
        context_repeated = context.unsqueeze(1).unsqueeze(1).repeat([1,self.n_nodes,self.n_nodes,1])

        # batch_size x from_nodes x to_nodes x self.edges_update_input_dim
        message_to_edge = torch.cat([all_influences,edge_influence,context_repeated],dim=-1)
        
        assert(len(message_to_edge.size())==4)
        assert(message_to_edge.size()[1]==self.n_nodes)
        assert(message_to_edge.size()[2]==self.n_nodes)
        return message_to_edge

    def message_collection_nodes(self, edge_influence, edges, nodes, context):
        # context = batch_size x context_length
        # edge_influence : batch_size x from_nodes x to_nodes x hidden_influence_dim
        
        masked_edge_influence = torch.mul(edge_influence,edges)

        # batch_size x nodes x hidden_influence_dim
        from_influence = masked_edge_influence.sum(dim=1)
        to_influence = masked_edge_influence.sum(dim=2)
        context_repeated = context.unsqueeze(1).repeat([1,self.n_nodes,1])
        # batch_size x nodes x hidden_influence_dim*2 + node_feature_len + context
        message_to_node = torch.cat([from_influence, to_influence, nodes, context_repeated],dim=-1)

        assert(len(message_to_node.size())==3)
        assert(message_to_node.size()[1]==self.n_nodes)
        return message_to_node
