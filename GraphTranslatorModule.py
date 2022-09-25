from importlib.util import module_for_loader
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
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

        self.evaluate = False

        self.cfg = model_configs
        self.n_nodes  = model_configs.n_nodes 
        self.n_len = model_configs.n_len
        self.c_len = model_configs.c_len
        self.n_activities = model_configs.n_activities
        self.edge_importance = model_configs.edge_importance
        self.context_dropout_probs = model_configs.context_dropout_probs ## change to context_dropout_probs
        self.learned_time_periods = model_configs.learned_time_periods
        self.preprocess_context = model_configs.preprocess_context
        self.contexts = list(self.context_dropout_probs.keys())

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

        self.embed_context_activity = nn.Linear(self.n_activities, self.c_len)
        self.embed_single_activity = lambda x: self.embed_context_activity(F.one_hot(x, num_classes=self.n_activities).to(torch.float32))


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

        self.mlp_predict_context = nn.Sequential(nn.Linear(self.edges_update_input_dim, self.hidden_influence_dim),
                                                    nn.ReLU(),
                                                    nn.Linear(self.hidden_influence_dim, self.c_len)
                                                    )


        self.mlp_update_nodes = nn.Sequential(nn.Linear(self.nodes_update_input_dim, self.hidden_influence_dim),
                                                    nn.ReLU(),
                                                    nn.Linear(self.hidden_influence_dim, 1)
                                                    )
        
        self.location_loss = lambda x,y: (nn.CrossEntropyLoss(reduction='none')(x.squeeze(-1).permute(0,2,1), y.squeeze(-1).long()))
        self.inference_location = lambda x: x.squeeze(-1).argmax(-1)

        self.class_loss = lambda xc,yc: nn.CrossEntropyLoss(reduction='none')(xc.permute(0,2,1), yc.long())
        self.inference_class = lambda xc: xc.argmax(-1)

        # self.context_loss = lambda xc, yc: torch.nn.CosineEmbeddingLoss()(xc, yc, torch.Tensor([1]).to('cuda'))
        self.activity_prediction_loss = lambda xc,yc: nn.CrossEntropyLoss(reduction='none')(xc.double(), yc.squeeze(-1).to(torch.long))

        # self.weighted_combination = nn.Linear(self.num_chebyshev_polys, 1, bias=False)

        ## transition encoder-decoder
        self.graph_encoder = geom_nn.Sequential('x, edge_index',
                                                [(geom_nn.GCNConv(self.n_len, self.c_len*4), 'x, edge_index -> x'),
                                                nn.ReLU(inplace=True),
                                                (geom_nn.GCNConv(self.c_len*4, self.c_len*2), 'x, edge_index -> x'),
                                                ])
        self.context_from_graph_encodings = nn.Sequential(nn.Linear(self.c_len*2, self.c_len*2),
                                                          nn.ReLU(),
                                                          nn.Linear(self.c_len*2, self.c_len)
                                                          )

    def context_loss(self, xc, yc):
        batch_size = xc.size()[0]
        xgrid, ygrid = torch.meshgrid(torch.arange(batch_size), torch.arange(batch_size), indexing='ij')
        same = (((xgrid==ygrid).to(int).reshape(-1)*2)-1).to('cuda')
        return torch.nn.CosineEmbeddingLoss(reduction='mean')(xc[xgrid.reshape(-1),:], yc[ygrid.reshape(-1),:], same)


    def graph_transition_encoder(self, nodes, edges, y_nodes, y_edges):
        batch_size, num_nodes, _ = nodes.size()
        mat2idx = lambda e_mat: (torch.argwhere(e_mat.reshape(batch_size*num_nodes, -1) == 1)).transpose(1,0)
        batch = torch.arange(batch_size).repeat(num_nodes,1).transpose(1,0).reshape(-1).to(int).to('cuda')
        graphs_in = geom_nn.global_mean_pool(self.graph_encoder(nodes.reshape(batch_size*num_nodes, -1), mat2idx(edges)), batch=batch)
        graphs_out = geom_nn.global_mean_pool(self.graph_encoder(y_nodes.reshape(batch_size*num_nodes, -1), mat2idx(y_edges)), batch=batch)
        context_diff = self.context_from_graph_encodings(graphs_out - graphs_in)
        assert context_diff.size()[0] == batch_size
        assert context_diff.size()[1] == self.c_len
        return context_diff

    def activity_encoder(self, activity):
        return self.embed_context_activity(F.one_hot(activity, num_classes=self.n_activities).to(torch.float32))

    def activity_decoder(self, latent_vector, ground_truth=None):
        activity_embeddings = self.embed_context_activity.weight
        assert activity_embeddings.size()[0] == self.c_len
        assert activity_embeddings.size()[1] == self.n_activities
        output_activity = torch.nn.CosineSimilarity(dim=1)(activity_embeddings.unsqueeze(0), latent_vector.unsqueeze(-1))
        if ground_truth is not None:
            activity_pred_loss = self.activity_prediction_loss(output_activity, ground_truth)
        else:
            activity_pred_loss = None
        output_activity = F.softmax(output_activity, dim=-1)
        return output_activity, activity_pred_loss

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

        return time_context.reshape(-1, self.c_len)

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
        xpool = xe.sum(dim=2).sum(dim=1)
        con = self.mlp_predict_context(xpool).view(size=[batch_size, self.c_len])

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


    def step(self, batch, prev_activity_probs=None):
        edges = batch['edges']
        nodes = batch['nodes']
        y_edges = batch['y_edges']
        y_nodes = batch['y_nodes']
        dyn_edges = batch['dynamic_edges_mask']
        batch_size = nodes.size()[0]
        
        time_context = self.get_time_context(batch['time'], batch['context_time'])
        activity_context = self.embed_single_activity(batch['activity'].to(int)).reshape(time_context.size())
        diff_context = self.graph_transition_encoder(nodes, edges, y_nodes, y_edges)
        
        context = time_context * 0
        if self.evaluate:
            if 'time' in self.contexts:
                context += time_context
            if prev_activity_probs is not None:
                context += self.embed_context_activity(prev_activity_probs).reshape(time_context.size())
            elif 'activity' in self.contexts and random() > self.cfg.context_dropout_probs['activity']:
                context += activity_context
        else:
            if 'time' in self.contexts and random() > self.cfg.context_dropout_probs['time']:
                context += time_context
            if 'activity' in self.contexts and random() > self.cfg.context_dropout_probs['activity']:
                context += activity_context
            if 'diff' in self.contexts and  random() > self.cfg.context_dropout_probs['diff']:
                context += diff_context
                
        context = torch.nn.functional.normalize(context)
  
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

        perc_time_contex_similarity = torch.nn.CosineSimilarity(dim=1)(time_context,diff_context)
        perc_act_contex_similarity = torch.nn.CosineSimilarity(dim=1)(activity_context, diff_context)
        perc_context_pred_similarity = torch.nn.CosineSimilarity(dim=1)(context_pred,activity_context)
        output_activity, activity_pred_loss = self.activity_decoder(context_pred, ground_truth = batch['y_activity'])

        logs = {'context time processing similarity': perc_time_contex_similarity,
                'context act processing similarity': perc_act_contex_similarity,
                'context prediction similarity': perc_context_pred_similarity,
                'processed activity context norm': torch.linalg.norm(activity_context),
                'processed time context norm': torch.linalg.norm(time_context),
                'processed diff norm': torch.linalg.norm(diff_context),
                'predicted context norm': torch.linalg.norm(context_pred),
                'Activity prediction accuracy': (output_activity.argmax(-1)==batch['y_activity']).sum()/len(batch['y_activity'])}

        losses = {'class':self.class_loss(output_probs['class'], gt['class']),
                  'location':self.location_loss(edges_pred, gt['location']),
                  'context_process':self.context_loss(time_context, diff_context) + self.context_loss(activity_context, diff_context),
                # 'context_predict':self.context_loss(context_pred, context_next)}
                  'context_predict':activity_pred_loss}

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
                   'logs':logs}

        return loss, details, output_activity

        
    def training_step(self, batch, batch_idx):
        loss, details, activity_probs_pred = self.step(batch, prev_activity_probs=None)
        self.log('Train loss',loss)
        self.log('Train',details['logs'])
        assert (sum(loss.values())).size() == torch.Size([])
        return sum(loss.values())

    def test_step(self, batch, batch_idx):
        loss, details, activity_probs_pred = self.step(batch, prev_activity_probs=None)
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
