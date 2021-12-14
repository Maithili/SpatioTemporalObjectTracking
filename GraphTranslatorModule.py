import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from pytorch_lightning.core.lightning import LightningModule

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

def _get_masks(gt_tensor, inference_tensor):
    masks = {}
    masks['tp'] = torch.logical_and(gt_tensor, inference_tensor)
    masks['fp'] = torch.logical_and(torch.logical_not(gt_tensor), inference_tensor)
    masks['fn'] = torch.logical_and(gt_tensor, torch.logical_not(inference_tensor))
    masks['tn'] = torch.logical_and(torch.logical_not(gt_tensor), torch.logical_not(inference_tensor))
    return masks

def _classification_metrics(gt_tensor, output_tensor, inference_tensor, loss_tensor):
    masks = _get_masks(gt_tensor, inference_tensor)
    result = {'eval':{} ,'losses':{}}
    result['eval']['accuracy'] = (masks['tp'].sum()+masks['tn'].sum())/torch.numel(gt_tensor)
    result['eval']['precision'] = (masks['tp'].sum())/(masks['fn'].sum() + (masks['tp'].sum()))
    result['eval']['recall'] = (masks['tp'].sum())/(masks['fp'].sum() + (masks['tp'].sum()))
    result['losses']['mean'] = loss_tensor.mean()
    result['losses']['tp'] = loss_tensor[masks['tp']].sum()/masks['tp'].sum()
    result['losses']['fp'] = loss_tensor[masks['fp']].sum()/masks['fp'].sum()
    result['losses']['fn'] = loss_tensor[masks['fn']].sum()/masks['fn'].sum()
    result['losses']['tn'] = loss_tensor[masks['tn']].sum()/masks['tn'].sum()
    return result

def _binary_metrics(input_tensor, output_tensor, inference_tensor):
    return {}

def evaluate(input, output, gt, losses, inferences):
    location_results = _classification_metrics(gt['location'], output['location'], inferences['location'], losses['location'])
    return {'location':location_results}

class GraphTranslatorModule(LightningModule):
    def __init__(self, 
                num_nodes, 
                node_feature_len,
                node_class_len,
                node_state_len,
                edge_feature_len, 
                context_len, 
                use_spectral_loss=True, 
                num_chebyshev_polys=2, 
                tree_formulation=False,
                node_accuracy_weight=0.5,
                learn_nodes=False):
        
        super().__init__()

        self.num_nodes  = num_nodes 
        self.node_feature_len = node_feature_len
        self.node_class_len = node_class_len
        self.node_state_len = node_state_len
        assert node_feature_len == node_class_len + node_state_len, f"Node class and state lengths should sum up to feature length, i.e. {node_class_len} + {node_state_len} == {node_feature_len}"
        self.edge_feature_len = edge_feature_len
        self.context_len = context_len
        self.node_accuracy_weight = node_accuracy_weight

        self.use_spectral_loss = use_spectral_loss
        self.num_chebyshev_polys = num_chebyshev_polys
        self.map_spectral_loss = 0

        self.hidden_influence_dim = 20
        
        self.mlp_influence = nn.Sequential(
                             nn.Linear(self.node_feature_len*2+self.edge_feature_len, 20),
                             nn.ReLU(),
                             nn.Linear(20, self.hidden_influence_dim),
                             )

        ## edge update layers
        self.mlp_update_edges = nn.Sequential(
                               nn.Linear(self.hidden_influence_dim*4+self.edge_feature_len+self.context_len, 20),
                               nn.ReLU(),
                               nn.Linear(20, self.edge_feature_len)
                               )
        self.accuracy_loss_edge = lambda x,y: (nn.CrossEntropyLoss(reduction='none')(x.squeeze(-1).permute(0,2,1), y.squeeze(-1).long()))
        self.inference_edge = lambda x: x.squeeze(-1).argmax(-1)

        ## node update layers
        self.mlp_update_nodes = nn.Sequential(
                               nn.Linear(self.hidden_influence_dim*2+self.node_feature_len+self.context_len, 20),
                               nn.ReLU(),
                               nn.Linear(20, self.node_feature_len)
                               )
        if learn_nodes:
            self.node_class_loss = lambda xc,yc: nn.CrossEntropyLoss(reduction='none')(xc.permute(0,2,1), yc.long())
            self.node_state_loss = lambda xs,ys: ((nn.MSELoss(reduction='none')(torch.tanh(xs), ys)) * torch.abs(ys)).sum(-1) / (torch.abs(ys)).sum(-1)
        else:
            self.node_class_loss = lambda xc,yc: torch.zeros_like(xc.sum(-1))
            self.node_state_loss = lambda xs,ys: torch.zeros_like(xs.sum(-1))
        self.inference_node_class = lambda xc: xc.argmax(-1)
        self.inference_node_state = lambda xs: torch.round(torch.tanh(xs)).to(int)

        self.weighted_combination = nn.Linear(self.num_chebyshev_polys, 1, bias=False)
        
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
        batch_size_e, num_f_nodes, num_t_nodes, edge_feature_len = edges.size()
        
        # Sanity check input dimensions
        assert batch_size == batch_size_e, "Different edge and node batch sizes"
        assert self.edge_feature_len == edge_feature_len, (str(self.edge_feature_len) +'!='+ str(edge_feature_len))
        assert self.node_feature_len == node_feature_len, (str(self.node_feature_len) +'!='+ str(node_feature_len))
        assert self.num_nodes == num_nodes, (str(self.num_nodes) +'!='+ str(num_nodes))
        assert self.num_nodes == num_f_nodes, (str(self.num_nodes) +'!='+ str(num_f_nodes))
        assert self.num_nodes == num_t_nodes, (str(self.num_nodes) +'!='+ str(num_t_nodes))

        x = self.collate_edges(edges=edges, nodes=nodes)
        x = x.view(
            size=[batch_size * self.num_nodes * self.num_nodes, 
                  self.node_feature_len * 2 + self.edge_feature_len])
        x = self.mlp_influence(x)
        x = x.view(
            size=[batch_size, 
                  self.num_nodes, 
                  self.num_nodes, 
                  self.hidden_influence_dim])

        ## edge update
        xe = self.message_collection_edges(x, edges, context)
        xe = xe.view(
            size=[batch_size * self.num_nodes * self.num_nodes, 
                  self.hidden_influence_dim*4 + self.edge_feature_len + self.context_len])
        xe = self.mlp_update_edges(xe).view(size=[batch_size, 
                                          self.num_nodes, 
                                          self.num_nodes, 
                                          self.edge_feature_len])

        ## node update
        xn = self.message_collection_nodes(x, nodes, context)
        xn = xn.view(
            size=[batch_size * self.num_nodes, 
                  self.hidden_influence_dim*2 + self.node_feature_len + self.context_len])
        xn = self.mlp_update_nodes(xn).view(size=[batch_size, 
                                          self.num_nodes, 
                                          self.node_feature_len])

        return xe, xn

    def step(self, batch):
        edges = batch['edges']
        nodes = batch['nodes']
        context = batch['context']
        y_edges = batch['y_edges']
        y_nodes = batch['y_nodes']
        
        edges_pred, nodes_pred = self(edges, nodes, context)

        input = {'class':nodes[:,:,:self.node_class_len], 'state':nodes[:,:,self.node_class_len:], 'location':self.inference_edge(edges)}
        output_probs = {'class':nodes_pred[:,:,:self.node_class_len], 'state':nodes_pred[:,:,self.node_class_len:], 'location':edges_pred}
        gt = {'class':y_nodes[:,:,:self.node_class_len], 'state':y_nodes[:,:,self.node_class_len:], 'location':self.inference_edge(y_edges)}
        losses = self.losses(output_probs, gt)
        inferences = self.inference(output_probs, gt)
        
        eval = evaluate(input, output_probs, gt, losses, inferences)

        return eval['location']


    def training_step(self, batch, batch_idx):
        eval = self.step(batch)
        self.log('Train performance',eval['eval'])
        self.log('Train losses',eval['losses'])
        return eval['losses']['mean'] + self.map_spectral_loss

    def test_step(self, batch, batch_idx):
        eval = self.step(batch)
        self.log('Test performance',eval['eval'])
        self.log('Test losses',eval['losses'])
        return 

    def losses(self, output, gt):
        location_losses = self.accuracy_loss_edge(output['location'], gt['location'])
        class_losses = self.node_class_loss(output['class'], gt['state'])
        state_losses = self.node_state_loss(output['class'], gt['state'])
        return {'location':location_losses, 'class':class_losses, 'state':state_losses}

    def inference(self, output, gt):
        location_inference = self.inference_edge(output['location'])
        class_inference = self.inference_node_class(output['class'])
        state_inference = self.inference_node_state(output['class'])
        return {'location':location_inference, 'class':class_inference, 'state':state_inference}

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
        assert(concatenated.size()[3]==self.node_feature_len*2+self.edge_feature_len)
        return concatenated

    def message_collection_edges(self, edge_influence, edges, context):
        # context = batch_size x context_length
        # edge_influence : batch_size x from_nodes x to_nodes x hidden_influence_dim

        # batch_size x nodes x 1 x hidden_influence_dim
        from_from_influence = edge_influence.sum(dim=2).unsqueeze(2).repeat([1,1,self.num_nodes,1])
        from_to_influence = edge_influence.sum(dim=1).unsqueeze(2).repeat([1,1,self.num_nodes,1])
        # batch_size x 1 x nodes x hidden_influence_dim
        to_to_influence = edge_influence.sum(dim=1).unsqueeze(1).repeat([1,self.num_nodes,1,1])
        to_from_influence = edge_influence.sum(dim=2).unsqueeze(1).repeat([1,self.num_nodes,1,1])
        
        # all_influences : batch_size x from_nodes x to_nodes x hidden_influence_dim
        all_influences = torch.cat([from_from_influence, from_to_influence, to_to_influence, to_from_influence],dim=-1)
        context_repeated = context.unsqueeze(1).unsqueeze(1).repeat([1,self.num_nodes,self.num_nodes,1])

        # batch_size x from_nodes x to_nodes x (hidden_influence_dim*4 + edge_feature_len + context_length)
        message_to_edge = torch.cat([all_influences,edges,context_repeated],dim=-1)
        assert(len(message_to_edge.size())==4)
        assert(message_to_edge.size()[1]==self.num_nodes)
        assert(message_to_edge.size()[2]==self.num_nodes)
        assert(message_to_edge.size()[3]==self.hidden_influence_dim*4+self.edge_feature_len+self.context_len)
        return message_to_edge

    def message_collection_nodes(self, edge_influence, nodes, context):
        # context = batch_size x context_length
        # edge_influence : batch_size x from_nodes x to_nodes x hidden_influence_dim

        from_influence = edge_influence.sum(dim=1)
        to_influence = edge_influence.sum(dim=2)
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
