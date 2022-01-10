import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from pytorch_lightning.core.lightning import LightningModule

def _erase_edges(edges):
    return edges*0 + (1/edges.size()[-1])

THRESH=0.5
def chebyshev_polynomials(edges, k):
    """
    Calculate Chebyshev polynomials up to order k.
    Result is (batch_size x k x N_nodes x N_nodes)
    """
    batch_size, n_nodes, _, _ = edges.size()

    # 1 if max value > thresh else 0 
    values, _ = edges.max(dim=-1)
    adj= torch.ceil(values - THRESH)

    # Laplacian applies to symmetric stuff only
    adj = (adj + adj.permute(0,2,1))/2
    degree_inv_sqrt = torch.diag_embed(torch.pow((adj.sum(dim=-1))+0.00001,-0.5))
    I_per_batch = torch.eye(n_nodes).unsqueeze(0).repeat([batch_size,1,1])
    laplacian = I_per_batch - torch.matmul(torch.matmul(degree_inv_sqrt, adj), degree_inv_sqrt)
    scaled_laplacian = 2/1.5*laplacian - I_per_batch

    polynomials = torch.cat([I_per_batch.unsqueeze(1), scaled_laplacian.unsqueeze(1)], dim=1)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_laplacian):
        next_poly = 2 * torch.matmul(scaled_laplacian,t_k_minus_one) - t_k_minus_two
        return next_poly

    for _ in range(2, k):
        next_poly =chebyshev_recurrence(polynomials[:,-1,:,:], polynomials[:,-2,:,:], scaled_laplacian)
        polynomials=torch.cat([polynomials,next_poly.unsqueeze(1)], dim=1)

    return polynomials

def _get_masks(gt_tensor, output_tensor):
    masks = {}
    masks['correct'] = gt_tensor == output_tensor
    masks['wrong'] = gt_tensor != output_tensor
    return masks

def _classification_metrics(gt_tensor, output_tensor, loss_tensor):
    masks = _get_masks(gt_tensor, output_tensor)
    result = {'accuracy':None ,'losses':{}}
    result['accuracy'] = (masks['correct'].sum())/torch.numel(gt_tensor)
    result['losses']['mean'] = loss_tensor.mean()
    result['losses']['correct'] = loss_tensor[masks['correct']].sum()/masks['correct'].sum()
    result['losses']['wrong'] = loss_tensor[masks['wrong']].sum()/masks['wrong'].sum()
    return result

def _binary_metrics(gtt_tensor, output_tensor, loss_tensor):
    return {}

def evaluate(gt, losses, output, evaluate_node):
    gt_tensor = gt['location'][evaluate_node]
    output_tensor = output['location'][evaluate_node]
    loss_tensor = losses['location'][evaluate_node]
    location_results = _classification_metrics(gt_tensor, output_tensor, loss_tensor)
    return {'location':location_results}

class GraphTranslatorModule(LightningModule):
    def __init__(self, 
                num_nodes, 
                node_feature_len,
                context_len, 
                node_accuracy_weight=0.5,
                learn_nodes=False,
                edge_importance=True):
        
        super().__init__()

        self.num_nodes  = num_nodes 
        self.node_feature_len = node_feature_len
        self.context_len = context_len
        self.node_accuracy_weight = node_accuracy_weight
        self.learn_nodes = learn_nodes
        self.edge_importance = edge_importance

        self.map_spectral_loss = 0

        self.hidden_influence_dim = 20

        self.edges_update_input_dim = self.hidden_influence_dim*4 + 1 + self.context_len
        
        self.mlp_influence    = {}     
        self.mlp_update_edges = {}        
        self.mlp_update_nodes = {}

        self.mlp_influence = nn.Sequential(nn.Linear(2*self.node_feature_len+1, 20),
                                                    nn.ReLU(),
                                                    nn.Linear(20, self.hidden_influence_dim),
                                                    )
        self.mlp_update_importance = nn.Sequential(nn.Linear(self.edges_update_input_dim, 20),
                                                    nn.ReLU(),
                                                    nn.Linear(20, 1)
                                                    )
        self.mlp_update_edges = nn.Sequential(nn.Linear(self.edges_update_input_dim, 20),
                                                    nn.ReLU(),
                                                    nn.Linear(20, 1)
                                                    )
        self.mlp_update_nodes = nn.Sequential(nn.Linear(self.hidden_influence_dim*2+self.node_feature_len+self.context_len, 20),
                                                    nn.ReLU(),
                                                    nn.Linear(20, self.node_feature_len)
                                                    )
        
        self.location_loss = lambda x,y: (nn.CrossEntropyLoss(reduction='none')(x.squeeze(-1).permute(0,2,1), y.squeeze(-1).long()))
        self.inference_location = lambda x: x.squeeze(-1).argmax(-1)

        self.class_loss = lambda xc,yc: nn.CrossEntropyLoss(reduction='none')(xc.permute(0,2,1), yc.long())
        self.inference_class = lambda xc: xc.argmax(-1)

        # self.weighted_combination = nn.Linear(self.num_chebyshev_polys, 1, bias=False)
        
    def graph_step(self, edges, nodes, context):

        batch_size, num_nodes, node_feature_len = nodes.size()
        
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

        if self.edge_importance:
            ## importance update
            imp = self.message_collection_edges(x, edges.unsqueeze(-1), context)
            imp = imp.view(
                size=[batch_size * self.num_nodes * self.num_nodes, 
                    self.edges_update_input_dim])
            imp = self.mlp_update_importance(imp).view(size=[batch_size, 
                                            self.num_nodes, 
                                            self.num_nodes,
                                            1])
        else:
            imp = edges.unsqueeze(-1)

        ## edge update
        xe = self.message_collection_edges(x, imp, context)
        xe = xe.view(
            size=[batch_size * self.num_nodes * self.num_nodes, 
                self.edges_update_input_dim])
        xe = self.mlp_update_edges(xe).view(size=[batch_size, 
                                        self.num_nodes, 
                                        self.num_nodes,
                                        1])

        ## node update
        if self.learn_nodes:
            xn = self.message_collection_nodes(torch.mul(x,imp), nodes, context)
            xn = xn.view(
                size=[batch_size * self.num_nodes, 
                    self.hidden_influence_dim*2 + self.node_feature_len + self.context_len])
            xn = self.mlp_update_nodes(xn).view(size=[batch_size, 
                                            self.num_nodes, 
                                            self.node_feature_len])
        else:
            xn = nodes        

        return xe, xn

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
        edges, nodes = self.graph_step(edges, nodes, context)

        return edges.squeeze(-1), nodes

    def step(self, batch):
        edges = batch['edges']
        nodes = batch['nodes']
        context = batch['context']
        y_edges = batch['y_edges']
        y_nodes = batch['y_nodes']
        dyn_edges = batch['dynamic_edges_mask']
        
        edges_pred, nodes_pred = self(edges, nodes, context)

        assert edges_pred.size() == dyn_edges.size(), f'Size mismatch in edges {edges_pred.size()} and dynamic mask {dyn_edges.size()}'
        edges_pred[dyn_edges == 0] = -float("inf")
        evaluate_node = dyn_edges.sum(-1) > 0

        input = {'class':self.inference_class(nodes), 
                 'location':self.inference_location(edges)}
                 
        output_probs = {'class':nodes_pred, 
                        'location':edges_pred}

        gt = {'class':self.inference_class(y_nodes), 
              'location':self.inference_location(y_edges)}

        losses = {'class':self.class_loss(output_probs['class'], gt['class']),
                  'location':self.location_loss(output_probs['location'], gt['location'])}

        output = {'class':self.inference_class(output_probs['class']),
                  'location':self.inference_location(output_probs['location'])}
        
        # for result, name in zip([input, gt, losses, output], ['input', 'gt', 'losses', 'output']):
        #     assert list(result['class'].size()) == list(nodes.size())[:2], 'wrong class size for {} : {} vs {}'.format(name, result['class'].size(), nodes.size())
        #     assert list(result['location'].size()) == list(nodes.size())[:2], 'wrong class size for {} : {} vs {}'.format(name, result['class'].size(), nodes.size())

        # assert list(output_probs['class'].size())[:-1] == list(nodes.size())[:-1], 'wrong class size for probs : {} vs {}'.format(output_probs['class'].size(), nodes.size())
        # assert list(output_probs['location'].size()) == list(edges.size()), 'wrong class size for probs : {} vs {}'.format(output_probs['class'].size(), edges.size())

        eval = evaluate(gt, losses, output, evaluate_node)

        return eval['location'], {'input':input, 'output_probs':output_probs, 'gt':gt, 'losses':losses, 'output':output, 'evaluate_node':evaluate_node}


    def training_step(self, batch, batch_idx):
        eval,_ = self.step(batch)
        self.log('Train accuracy',eval['accuracy'])
        self.log('Train losses',eval['losses'])
        return eval['losses']['mean'] + self.map_spectral_loss

    def test_step(self, batch, batch_idx):
        eval, details = self.step(batch)
        self.log('Test accuracy',eval['accuracy'])
        self.log('Test losses',eval['losses'])
        
        uncond_batch = batch
        uncond_batch['edges'] = _erase_edges(uncond_batch['edges'])
        eval, details = self.step(uncond_batch)
        self.log('Test accuracy (Unconditional)',eval['accuracy'])
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
        message_to_edge = torch.cat([all_influences,edges,context_repeated],dim=-1)
        
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

    def graph_regularized_spectral_loss(self, edges, nodes):
        # batch_size x k x N_nodes x N_nodes
        chebyshev_polys = chebyshev_polynomials(edges, self.num_chebyshev_polys)
        flattened_polys = chebyshev_polys.permute([0,2,3,1]).reshape(-1,self.num_chebyshev_polys)
        combined_polys = self.weighted_combination(flattened_polys).reshape(-1, self.num_nodes, self.num_nodes)
        assert edges.size()[0] == combined_polys.size()[0], 'Chebyshev poly combination gone wrong :( '
        spectral_loss = (torch.matmul(combined_polys, nodes)).square().mean()
        regularization = abs(1 - sum(p.pow(2.0).sum() for p in self.weighted_combination.parameters()))
        return (1/3) * spectral_loss + (0.01) * regularization
