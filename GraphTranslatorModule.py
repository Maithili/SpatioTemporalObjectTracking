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


class GraphTranslatorModule(LightningModule):
    def __init__(self, 
                num_nodes, 
                node_feature_len,
                node_class_len,
                node_state_len,
                edge_feature_len, 
                context_len, 
                output_filters, 
                use_spectral_loss=True, 
                num_chebyshev_polys=2, 
                allow_multiple_edge_types=False,
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
        self.output_filters = output_filters
        self.allow_multiple_edge_types = allow_multiple_edge_types
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
        if allow_multiple_edge_types:
            self.mlp_update_edges = nn.Sequential(
                               nn.Linear(self.hidden_influence_dim*4+self.edge_feature_len+self.context_len, 20),
                               nn.ReLU(),
                               nn.Linear(20, self.edge_feature_len),
                               nn.Sigmoid()
                               )
            self.accuracy_loss_edge = nn.BCELoss(reduction='none')
            self.inference_accuracy_edge = lambda x,y: ((x > 0.5).to(dtype=int) == y).to(dtype=float)
        
        else:
            self.mlp_update_edges = nn.Sequential(
                               nn.Linear(self.hidden_influence_dim*4+self.edge_feature_len+self.context_len, 20),
                               nn.ReLU(),
                               nn.Linear(20, self.edge_feature_len)
                               )
            self.accuracy_loss_edge = lambda x,y: (nn.CrossEntropyLoss(reduction='none')(x.permute(0,3,1,2), y.argmax(-1).long())).unsqueeze(-1)
            self.inference_accuracy_edge = lambda x,y: (x.argmax(-1) == y.argmax(-1)).to(dtype=float).unsqueeze(-1)

        ## node update layers
        self.mlp_update_nodes = nn.Sequential(
                               nn.Linear(self.hidden_influence_dim*2+self.node_feature_len+self.context_len, 20),
                               nn.ReLU(),
                               nn.Linear(20, self.node_feature_len)
                               )
        self.node_class_loss = lambda xc,yc: nn.CrossEntropyLoss(reduction='none')(xc.permute(0,2,1), yc.argmax(-1).long())
        self.node_state_loss = lambda xs,ys: ((nn.MSELoss(reduction='none')(torch.tanh(xs), ys)) * torch.abs(ys)).sum(-1) / (torch.abs(ys)).sum(-1)
        if learn_nodes:
            self.inference_accuracy_node_class = lambda xc,yc: (xc.argmax(-1) == yc.argmax(-1)).to(dtype=float)
            self.inference_accuracy_node_state = lambda xs,ys: ((torch.round(torch.tanh(xs)).to(int) == ys.to(int)).to(dtype=float) * torch.abs(ys)).sum(-1) / (torch.abs(ys)).sum(-1)
        else:
            self.inference_accuracy_node_class = lambda xc,yc: torch.zeros_like(xc.sum(-1))
            self.inference_accuracy_node_state = lambda xs,ys: torch.zeros_like(xs.sum(-1))

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
        y_edges = batch['y_edges']
        y_nodes = batch['y_nodes']
        
        edges_pred, nodes_pred = self(edges, nodes, context)
        edge_losses, node_class_losses, node_state_losses = self.losses(edges_pred, nodes_pred, y_edges, y_nodes, nodes)
        
        self.output_filters.set_data_info(x_edges = edges.argmax(-1).unsqueeze(-1), pred_edges = edges_pred.argmax(-1).unsqueeze(-1), y_edges = y_edges.argmax(-1).unsqueeze(-1), node_classes=nodes[:,:,:self.node_class_len])
        self.log("Train loss", self.output_filters.logging_metrics(edge_losses, node_class_losses, node_state_losses))

        return self.output_filters.train_metric(edge_losses, node_class_losses+node_state_losses) + self.map_spectral_loss

    def test_step(self, batch, batch_idx):
        edges = batch['edges']
        nodes = batch['nodes']
        context = batch['context']
        y_edges = batch['y_edges']
        y_nodes = batch['y_nodes']

        edges_pred, nodes_pred = self(edges, nodes, context)
        edge_accuracy, node_class_accuracy, node_state_accuracy = self.inference_accuracy(edges_pred, nodes_pred, y_edges, y_nodes)

        self.output_filters.set_data_info(x_edges=edges.argmax(-1).unsqueeze(-1), pred_edges = edges_pred.argmax(-1).unsqueeze(-1), y_edges=y_edges.argmax(-1).unsqueeze(-1), node_classes=nodes[:,:,:self.node_class_len])
        self.log("Test accuracy", self.output_filters.logging_metrics(edge_accuracy, node_class_accuracy, node_state_accuracy))

        return self.output_filters.train_metric(edge_accuracy, node_class_accuracy+node_state_accuracy) + self.map_spectral_loss

    def losses(self, edges_pred, nodes_pred, edges_actual, nodes_actual, nodes_in):
        edge_losses = self.accuracy_loss_edge(edges_pred, edges_actual)
        node_class_losses = self.node_class_loss(nodes_pred[:,:,:self.node_class_len], nodes_actual[:,:,:self.node_class_len])
        node_state_losses = self.node_state_loss(nodes_pred[:,:,self.node_class_len:], nodes_actual[:,:,self.node_class_len:])
        if self.use_spectral_loss:
            self.map_spectral_loss = self.graph_regularized_spectral_loss(edges_pred, nodes_in)
        return edge_losses, node_class_losses, node_state_losses

    def inference_accuracy(self, edges_pred, nodes_pred, edges_actual, nodes_actual):
        edge_accuracy = self.inference_accuracy_edge(edges_pred,edges_actual)
        node_class_accuracy = self.inference_accuracy_node_class(nodes_pred[:,:,:self.node_class_len],nodes_actual[:,:,:self.node_class_len])
        node_state_accuracy = self.inference_accuracy_node_state(nodes_pred[:,:,self.node_class_len:],nodes_actual[:,:,self.node_class_len:])
        return edge_accuracy, node_class_accuracy, node_state_accuracy

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
