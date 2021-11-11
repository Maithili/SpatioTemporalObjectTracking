import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from pytorch_lightning.core.lightning import LightningModule
from analyzers import MeanLoss

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


class GraphTranslatorModule(LightningModule):
    def __init__(self, num_nodes, node_feature_len, edge_feature_len, context_len, train_analyzer=MeanLoss(), logging_analyzers=[], use_spectral_loss=True, num_chebyshev_polys=2):
        super().__init__()

        self.num_nodes  = num_nodes 
        self.node_feature_len = node_feature_len
        self.edge_feature_len = edge_feature_len
        self.context_len = context_len

        self.train_analyzer = train_analyzer
        self.logging_analyzers = logging_analyzers
        self.use_spectral_loss = use_spectral_loss
        self.num_chebyshev_polys = num_chebyshev_polys
        self.map_spectral_loss = 0

        self.hidden_influence_dim = 20
        
        self.mlp_influence = nn.Sequential(
                             nn.Linear(self.node_feature_len*2+self.edge_feature_len, 20),
                             nn.ReLU(),
                             nn.Linear(20, self.hidden_influence_dim),
                             )

        self.mlp_update = nn.Sequential(
                               nn.Linear(self.hidden_influence_dim*4+self.edge_feature_len+self.context_len, 20),
                               nn.ReLU(),
                               nn.Linear(20, self.edge_feature_len),
                               nn.Sigmoid()
                               )

        self.bce = nn.BCELoss(reduction='none')

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
        x = self.message_collection(x, edges, context)
        x = x.view(
            size=[batch_size * self.num_nodes * self.num_nodes, 
                  self.hidden_influence_dim*4 + self.edge_feature_len + self.context_len])
        x = self.mlp_update(x).view(size=[batch_size, 
                                          self.num_nodes, 
                                          self.num_nodes, 
                                          self.edge_feature_len])
        return x

    def training_step(self, batch, batch_idx):
        """
        Args:
            batch: x,y (tuple) 
                    x: adjacency,edges,nodes,context (tuple) : edges is nxnx_; nodes is nx_; context is any vector
                    y: edge_existence,edge_category : existence is 0/1; type is integer in [0,len(edge_feat))
        """
        edges = batch['edges']
        nodes = batch['nodes']
        context = batch['context']
        y = batch['y']

        x = self(edges, nodes, context)
        losses = self.losses(x, y, nodes)
        
        for analyzer in self.logging_analyzers:
            self.log('Train: '+analyzer.name(), analyzer(losses, x_edges=edges, y_edges=y, nodes=nodes))
        self.log('Spectral Loss: ',self.map_spectral_loss)

        return self.train_analyzer(losses, x_edges=edges, y_edges=y, nodes=nodes) + self.map_spectral_loss

    def test_step(self, batch, batch_idx):
        edges = batch['edges']
        nodes = batch['nodes']
        context = batch['context']
        y = batch['y']

        x = self(edges, nodes, context)
        losses = self.losses(x, y, nodes)

        for analyzer in self.logging_analyzers:
            self.log('Test: '+analyzer.name(), analyzer(losses, x_edges=edges, y_edges=y, nodes=nodes))

    def losses(self, x, y, nodes=None):
        losses = self.bce(x, y)
        if self.use_spectral_loss:
            self.map_spectral_loss = self.graph_regularized_spectral_loss(x, nodes)
        return losses

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

    def message_collection(self, edge_influence, edges, context):
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

    def graph_regularized_spectral_loss(self, edges, nodes):
        # batch_size x k x N_nodes x N_nodes
        chebyshev_polys = chebyshev_polynomials(edges, self.num_chebyshev_polys)
        flattened_polys = chebyshev_polys.permute([0,2,3,1]).reshape(-1,self.num_chebyshev_polys)
        combined_polys = self.weighted_combination(flattened_polys).reshape(-1, self.num_nodes, self.num_nodes)
        assert edges.size()[0] == combined_polys.size()[0], 'Chebyshev poly combination gone wrong :( '
        spectral_loss = (torch.matmul(combined_polys, nodes)).square().mean()
        regularization = abs(1 - sum(p.pow(2.0).sum() for p in self.weighted_combination.parameters()))
        return (1/3) * spectral_loss + (0.01) * regularization
