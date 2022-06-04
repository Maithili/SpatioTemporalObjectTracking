import sys
sys.path.append('helpers')
from random import random
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from pytorch_lightning.core.lightning import LightningModule

from breakdown_evaluations import _erase_edges


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

def evaluate_accuracy(gt_tensor, loss_tensor, output_tensor, input_tensor, tn_loss_weight):
    masks = get_masks(gt_tensor, output_tensor, input_tensor)
    result = {}
    result['accuracy'] = (masks['correct'].sum())/torch.numel(gt_tensor)
    if loss_tensor is not None:
        result['losses'] = {}
        if tn_loss_weight is not None:
            not_tn = np.bitwise_not(masks['tn']).to(bool)
            # important_losses = loss_tensor[cm_masks['tp']]
            # unimportant_losses = loss_tensor[np.bitwise_or(cm_masks['fp'], cm_masks['fn'])]
            important_losses = loss_tensor[not_tn]
            unimportant_losses = loss_tensor[masks['tn']]
            result['losses']['mean'] = (1 - tn_loss_weight) * important_losses.mean() + tn_loss_weight * unimportant_losses.mean()
            result['losses']['important'] = important_losses.mean()
            result['losses']['unimportant'] = unimportant_losses.mean()
        else:
            result['losses']['mean'] = loss_tensor.mean()
        result['losses']['correct'] = loss_tensor[masks['correct']].sum()/masks['correct'].sum()
        result['losses']['wrong'] = loss_tensor[masks['wrong']].sum()/masks['wrong'].sum()
    return result

def evaluate(gt, output, input, evaluate_node, losses=None, tn_loss_weight=None):
    gt_tensor = gt[evaluate_node]
    input_tensor = input[evaluate_node]
    output_tensor = output[evaluate_node]
    if losses is not None:
        loss_tensor = losses[evaluate_node]
        result = evaluate_accuracy(gt_tensor, loss_tensor, output_tensor, input_tensor, tn_loss_weight)
    else:
        result = evaluate_accuracy(gt_tensor, None, output_tensor, input_tensor, tn_loss_weight)
    # result['CM'] = evaluate_precision_recall(gt_tensor, output_tensor, input_tensor)
    return result


class GraphTranslatorModule(LightningModule):
    def __init__(self, 
                num_nodes, 
                node_feature_len,
                context_len, 
                edge_importance,
                edge_dropout_prob,
                tn_loss_weight,
                learned_time_periods,
                hidden_layer_size):
        
        super().__init__()

        self.num_nodes  = num_nodes 
        self.node_feature_len = node_feature_len
        self.context_len = context_len
        self.edge_importance = edge_importance
        self.edge_dropout_prob = edge_dropout_prob
        self.tn_loss_weight = tn_loss_weight
        self.learned_time_periods = learned_time_periods

        self.hidden_influence_dim = 20

        self.edges_update_input_dim = self.hidden_influence_dim*5 + self.context_len
        
        if learned_time_periods:
            self.period_in_days = torch.nn.Parameter(torch.randn((1, 3)))
            self.context_len = 2*3
            omega_one_day = torch.Tensor([2*np.pi/60*24])
            self.context_from_time = lambda t : torch.cat((torch.cos(omega_one_day * t / self.period_in_days),torch.sin(omega_one_day * t / self.period_in_days)), axis=1)
 
        mlp_hidden = hidden_layer_size

        self.mlp_influence = nn.Sequential(nn.Linear(2*self.node_feature_len+1, mlp_hidden),
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

        # self.weighted_combination = nn.Linear(self.num_chebyshev_polys, 1, bias=False)
        
    def graph_step(self, edges, nodes, context):

        batch_size, num_nodes, node_feature_len = nodes.size()

        context = context.view(size=[batch_size, self.context_len])

        x = self.collate_edges(edges=edges.unsqueeze(-1), nodes=nodes)
        x = x.view(
            size=[batch_size * self.num_nodes * self.num_nodes, 
                  2*self.node_feature_len+1])
        x = self.mlp_influence(x)
        x = x.view(
            size=[batch_size, 
                  self.num_nodes, 
                  self.num_nodes, 
                  self.hidden_influence_dim])

        if self.edge_importance == 'predicted':
            ## importance update
            imp = self.message_collection_edges(x, edges.unsqueeze(-1), context)
            imp = imp.view(
                size=[batch_size * self.num_nodes * self.num_nodes, 
                    self.edges_update_input_dim])
            imp = self.mlp_update_importance(imp).view(size=[batch_size, 
                                            self.num_nodes, 
                                            self.num_nodes,
                                            1])
        elif self.edge_importance == 'all':
            imp = torch.ones_like(edges.unsqueeze(-1))
        elif self.edge_importance == 'existing':
            imp = edges.unsqueeze(-1)
        else:
            raise KeyError(f'Edge Importance given as ({self.edge_importance}) is not among predicted, all or existing')

        ## edge update
        xe = self.message_collection_edges(x, imp, context)
        xe = xe.view(
            size=[batch_size * self.num_nodes * self.num_nodes, 
                self.edges_update_input_dim])
        xe = self.mlp_update_edges(xe).view(size=[batch_size, 
                                        self.num_nodes, 
                                        self.num_nodes,
                                        1])
        
        return xe, nodes, imp.view(size=[batch_size, self.num_nodes, self.num_nodes]).detach().cpu().numpy()

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
        assert self.node_feature_len == node_feature_len, (str(self.node_feature_len) +'!='+ str(node_feature_len))
        assert self.num_nodes == num_nodes, (str(self.num_nodes) +'!='+ str(num_nodes))
        assert self.num_nodes == num_f_nodes, (str(self.num_nodes) +'!='+ str(num_f_nodes))
        assert self.num_nodes == num_t_nodes, (str(self.num_nodes) +'!='+ str(num_t_nodes))

        # for step in range(len(self.edge_feature_len)-1):
        edges, nodes, imp = self.graph_step(edges, nodes, context)

        return edges.squeeze(-1), nodes, imp

    def step(self, batch):
        edges = batch['edges']
        nodes = batch['nodes']
        y_edges = batch['y_edges']
        y_nodes = batch['y_nodes']
        dyn_edges = batch['dynamic_edges_mask']
        
        if self.learned_time_periods:
            time = batch['time'].unsqueeze(1)
            context = self.context_from_time(time)
        else:
            context = batch['context']
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
                  'location':self.location_loss(edges_pred, gt['location'])}

        output = {'class':self.inference_class(output_probs['class']),
                  'location':self.inference_location(edges_inferred)}
        

        # for result, name in zip([input, gt, losses, output], ['input', 'gt', 'losses', 'output']):
        #     assert list(result['class'].size()) == list(nodes.size())[:2], 'wrong class size for {} : {} vs {}'.format(name, result['class'].size(), nodes.size())
        #     assert list(result['location'].size()) == list(nodes.size())[:2], 'wrong class size for {} : {} vs {}'.format(name, result['class'].size(), nodes.size())

        # assert list(output_probs['class'].size())[:-1] == list(nodes.size())[:-1], 'wrong class size for probs : {} vs {}'.format(output_probs['class'].size(), nodes.size())
        # assert list(output_probs['location'].size()) == list(edges.size()), 'wrong class size for probs : {} vs {}'.format(output_probs['class'].size(), edges.size())

        eval = evaluate(gt=gt['location'], output=output['location'], input=input['location'], evaluate_node=evaluate_node, losses=losses['location'], tn_loss_weight=self.tn_loss_weight)

        ## NOT USED
        # eval['duplication_loss'] = (F.softmax(output_probs['location']) * edges)[evaluate_node].sum()
        
        details = {'input':input, 
                   'output_probs':output_probs, 
                   'gt':gt, 
                   'losses':losses, 
                   'output':output, 
                   'evaluate_node':evaluate_node,
                   'importance_weights':imp}

        return eval, details


    def training_step(self, batch, batch_idx):
        dropout = False
        if random() < self.edge_dropout_prob:
            dropout = True
            batch['edges'] = _erase_edges(batch['edges'], batch['dynamic_edges_mask'])
        eval,_ = self.step(batch)
        self.log('Train accuracy',eval['accuracy'])
        self.log('Train losses',eval['losses'])
        return eval['losses']['mean']

    def test_step(self, batch, batch_idx):
        eval, details = self.step(batch)
        self.log('Test losses',eval['losses'])
        
        uncond_batch = batch
        uncond_batch['edges'] = _erase_edges(uncond_batch['edges'], batch['dynamic_edges_mask'])
        eval, details = self.step(uncond_batch)
        self.log('Test losses (Unconditional)',eval['losses'])

        return 

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def collate_edges(self, edges, nodes):
        # nodes_repeated : batch_size x nodes x repeat dimension x node_feature_len
        nodes_repeated = nodes.unsqueeze(2).repeat([1,1,self.num_nodes,1])
        # concatenated : batch_size x from_nodes x to_nodes x (node_feature * 2 + edge_feature)
        concatenated = torch.cat([nodes_repeated, nodes_repeated.permute(0,2,1,3), edges], dim=-1)
        assert(len(concatenated.size())==4)
        assert(concatenated.size()[1]==self.num_nodes)
        assert(concatenated.size()[2]==self.num_nodes)
        assert(concatenated.size()[3]==self.node_feature_len*2+1)
        return concatenated

    def message_collection_edges(self, edge_influence, edges, context):
        # context = batch_size x context_length
        # edge_influence : batch_size x from_nodes x to_nodes x hidden_influence_dim

        masked_edge_influence = torch.mul(edge_influence,edges)

        # batch_size x nodes x 1 x hidden_influence_dim
        from_from_influence = masked_edge_influence.sum(dim=2).unsqueeze(2).repeat([1,1,self.num_nodes,1])
        from_to_influence = masked_edge_influence.sum(dim=1).unsqueeze(2).repeat([1,1,self.num_nodes,1])
        # batch_size x 1 x nodes x hidden_influence_dim
        to_to_influence = masked_edge_influence.sum(dim=1).unsqueeze(1).repeat([1,self.num_nodes,1,1])
        to_from_influence = masked_edge_influence.sum(dim=2).unsqueeze(1).repeat([1,self.num_nodes,1,1])
        
        # all_influences : batch_size x from_nodes x to_nodes x hidden_influence_dim
        all_influences = torch.cat([from_from_influence, from_to_influence, to_to_influence, to_from_influence],dim=-1)
        context_repeated = context.unsqueeze(1).unsqueeze(1).repeat([1,self.num_nodes,self.num_nodes,1])

        # batch_size x from_nodes x to_nodes x self.edges_update_input_dim
        message_to_edge = torch.cat([all_influences,edge_influence,context_repeated],dim=-1)
        
        assert(len(message_to_edge.size())==4)
        assert(message_to_edge.size()[1]==self.num_nodes)
        assert(message_to_edge.size()[2]==self.num_nodes)
        return message_to_edge

    def message_collection_nodes(self, masked_edge_influence, nodes, context):
        # context = batch_size x context_length
        # edge_influence : batch_size x from_nodes x to_nodes x hidden_influence_dim

        from_influence = masked_edge_influence.sum(dim=1)
        to_influence = masked_edge_influence.sum(dim=2)
        context_repeated = context.unsqueeze(1).repeat([1,self.num_nodes,1])
        message_to_node = torch.cat([from_influence, to_influence, nodes, context_repeated],dim=-1)
        return message_to_node
