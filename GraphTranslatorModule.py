import os
import json
import sys
sys.path.append('helpers')
from random import random
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from pytorch_lightning.core.lightning import LightningModule

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
 
        # self.mlp_context = nn.Sequential(nn.Linear(self.c_len, self.c_len),
        #                                      nn.ReLU(),
        #                                      nn.Linear(self.c_len, self.c_len),
        #                                      )

        # print(model_configs.n_activities, self.c_len)
        # self.embed_context_action = torch.nn.Embedding(model_configs.n_activities, self.c_len)



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

        # self.mlp_predict_context = nn.Sequential(nn.Linear(self.edges_update_input_dim, self.hidden_influence_dim),
        #                                             nn.ReLU(),
        #                                             nn.Linear(self.hidden_influence_dim, self.c_len)
        #                                             )


        # self.mlp_update_nodes = nn.Sequential(nn.Linear(self.nodes_update_input_dim, self.hidden_influence_dim),
        #                                             nn.ReLU(),
        #                                             nn.Linear(self.hidden_influence_dim, 1)
        #                                             )
        
        self.location_loss = lambda x,y: (nn.CrossEntropyLoss(reduction='none')(x.squeeze(-1).permute(0,2,1), y.squeeze(-1).long()))
        self.inference_location = lambda x: x.squeeze(-1).argmax(-1)

        self.class_loss = lambda xc,yc: nn.CrossEntropyLoss(reduction='none')(xc.permute(0,2,1), yc.long())
        self.inference_class = lambda xc: xc.argmax(-1)

        self.context_loss = lambda context_list: sum([torch.nn.CosineEmbeddingLoss()(con, context_list[0], torch.Tensor([1]).to('cuda')) for con in context_list])

        # self.weighted_combination = nn.Linear(self.num_chebyshev_polys, 1, bias=False)

    def inference_location_comparative(self, probs, ref, thresh=0.2):
        non_comp_inf = F.one_hot(self.inference_location(probs))
        ref_onehot = F.one_hot(ref)
        not_good_enough = ((probs[non_comp_inf] - probs[ref_onehot]) < thresh).sum(-1)
        non_comp_inf[not_good_enough] = ref[not_good_enough]
        return non_comp_inf

    def get_time_context(self, batch_in):
        if self.learned_time_periods:
            input('Are you really trying to learn time periods?')
            time = batch_in['time'].unsqueeze(1)
            time_context = self.context_from_time(time)
        else:
            time_context = batch_in['context_time']
        
        # time_context = self.mlp_context(time_context)

        return time_context

    # def get_activity_context(self, batch_in):
    #     activity_context = self.embed_context_action(batch_in['context_activity'].to(int))
    #     assert activity_context.size()[-1] == self.c_len, f"Activity context size is {activity_context.size()} and clen is {self.c_len}"
    #     return activity_context


    def graph_step(self, edges, nodes, context):

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
        
        ## context prediction
        # xpool = xe.sum(dim=2).sum(dim=1)
        # con = self.mlp_predict_context(xpool).view(size=[batch_size, self.c_len])
        con = context

        ## edge update
        xe = xe.view(
            size=[batch_size * self.n_nodes * self.n_nodes, 
                self.edges_update_input_dim])
        xe = self.mlp_update_edges(xe).view(size=[batch_size, 
                                                self.n_nodes, 
                                                self.n_nodes,
                                                1])


        # ## node update
        # xn = self.message_collection_nodes(x, imp, nodes, context)
        # xn = xn.view(
        #     size=[batch_size * self.num_nodes * self.num_nodes, 
        #         self.edges_update_input_dim])
        # xn = self.mlp_update_nodes(xn).view(size=[batch_size, 
        #                                 self.num_nodes, 
        #                                 self.num_nodes,
        #                                 1])
        
        return xe, nodes, con, imp.view(size=[batch_size, self.n_nodes, self.n_nodes])

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

        # for step in range(len(self.edge_feature_len)-1):
        edges, nodes, context, imp = self.graph_step(edges, nodes, context)

        return edges.squeeze(-1), nodes, context, imp


    def step(self, batch, prev_context=None):
        edges = batch['edges']
        nodes = batch['nodes']
        y_edges = batch['y_edges']
        y_nodes = batch['y_nodes']
        dyn_edges = batch['dynamic_edges_mask']
        batch_size = nodes.size()[0]
        
        time_context = self.get_time_context(batch_in=batch)
        # activity_context = self.get_activity_context(batch_in=batch).reshape(time_context.size())
        # context = time_context * 0
        # if random() > self.cfg.context_dropout_probs['time']:
        #     context += time_context
        # if random() > self.cfg.context_dropout_probs['activity']:
        #     context += activity_context
        # # if random() > self.cfg.reset_prob and prev_context is not None:
        # if prev_context is not None:
        #     context += prev_context 
                
        # context = torch.nn.functional.normalize(context)
  
        context = time_context

        edges_pred, nodes_pred, context_pred, imp = self(edges, nodes, context)

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

        # if prev_context is None:
        #     context_list = [self.get_activity_context(batch_in=batch).squeeze(1), self.get_time_context(batch_in=batch).squeeze(1)]
        # else:
        #     context_list = [self.get_activity_context(batch_in=batch).squeeze(1), self.get_time_context(batch_in=batch).squeeze(1), prev_context.squeeze(1)]

        context_list = [time_context]

        losses = {'class':self.class_loss(output_probs['class'], gt['class']),
                  'location':self.location_loss(edges_pred, gt['location']),
                  'context':self.context_loss(context_list)}

        output = {'class':self.inference_class(output_probs['class']),
                  'location':self.inference_location(edges_inferred)}
        

        # for result, name in zip([input, gt, losses, output], ['input', 'gt', 'losses', 'output']):
        #     assert list(result['class'].size()) == list(nodes.size())[:2], 'wrong class size for {} : {} vs {}'.format(name, result['class'].size(), nodes.size())
        #     assert list(result['location'].size()) == list(nodes.size())[:2], 'wrong class size for {} : {} vs {}'.format(name, result['class'].size(), nodes.size())

        # assert list(output_probs['class'].size())[:-1] == list(nodes.size())[:-1], 'wrong class size for probs : {} vs {}'.format(output_probs['class'].size(), nodes.size())
        # assert list(output_probs['location'].size()) == list(edges.size()), 'wrong class size for probs : {} vs {}'.format(output_probs['class'].size(), edges.size())

        # eval = evaluate(gt=gt['location'], output=output['location'], input=input['location'], evaluate_node=evaluate_node, losses=losses['location'])
        # loss = {'graph': 1 * losses['location'][evaluate_node].mean(), 'context': 5 * losses['context'].mean()}
        loss = {'graph' : losses['location'][evaluate_node].mean()}

        details = {'input':input, 
                   'output_probs':output_probs, 
                   'gt':gt, 
                   'losses':losses, 
                   'output':output, 
                   'evaluate_node':evaluate_node,
                   'importance_weights':imp}

        return loss, details, context_pred

        
    def training_step(self, batch, batch_idx):
        prev_context = None
        loss, details, prev_context = self.step(batch, prev_context=None)
        self.log('Train loss',loss)
        return sum(loss.values())

    def test_step(self, batch, batch_idx):
        prev_context = None
        loss, details, prev_context = self.step(batch, prev_context=None)
        self.log('Test loss',loss)
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
