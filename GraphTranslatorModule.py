import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from pytorch_lightning.core.lightning import LightningModule
from analyzers import MeanLoss


class GraphTranslatorModule(LightningModule):
    def __init__(self, num_nodes, node_feature_len, edge_feature_len, context_len, train_analyzer=MeanLoss(), logging_analyzers=[]):
        super().__init__()

        self.num_nodes  = num_nodes 
        self.node_feature_len = node_feature_len
        self.edge_feature_len = edge_feature_len
        self.total_context_len = 2 * context_len

        self.train_analyzer = train_analyzer
        self.logging_analyzers = logging_analyzers

        self.hidden_influence_dim = 20
        
        self.mlp_influence = nn.Sequential(
                             nn.Linear(self.node_feature_len*2+self.edge_feature_len, 20),
                             nn.ReLU(),
                             nn.Linear(20, self.hidden_influence_dim),
                             )

        self.mlp_update = nn.Sequential(
                               nn.Linear(self.hidden_influence_dim+self.edge_feature_len+self.total_context_len, 20),
                               nn.ReLU(),
                               nn.Linear(20, self.edge_feature_len),
                               nn.Sigmoid()
                               )

        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, edges, nodes, context_curr, context_query):
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

        context = torch.cat([context_curr, context_query], dim=-1)

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
                  self.hidden_influence_dim + self.edge_feature_len + self.total_context_len])
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
        edges, nodes, context_curr, context_query, y = batch
        x = self(edges, nodes, context_curr, context_query)
        losses = self.bce(x, y)
        for analyzer in self.logging_analyzers:
            self.log('Train: '+analyzer.name(), analyzer(losses, x_edges=edges, y_edges=y))
        return self.train_analyzer(losses, x_edges=edges, y_edges=y)

    def test_step(self, batch, batch_idx):
        edges, nodes, context_curr, context_query, y = batch
        x = self(edges, nodes, context_curr, context_query)
        losses = self.bce(x, y)
        for analyzer in self.logging_analyzers:
            self.log('Test: '+analyzer.name(), analyzer(losses, x_edges=edges, y_edges=y))

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
        from_from_influence = edge_influence.sum(dim=2)
        from_to_influence = edge_influence.sum(dim=1)
        from_influence_sum_repeated = (from_from_influence + from_to_influence).unsqueeze(2).repeat([1,1,self.num_nodes,1])
        # batch_size x 1 x nodes x hidden_influence_dim
        to_to_influence = edge_influence.sum(dim=1)
        to_from_influence = edge_influence.sum(dim=2)
        to_influence_sum_repeated = (to_to_influence + to_from_influence).unsqueeze(1).repeat([1,self.num_nodes,1,1])
        
        # all_influences : batch_size x from_nodes x to_nodes x hidden_influence_dim
        all_influences = from_influence_sum_repeated + to_influence_sum_repeated
        context_repeated = context.unsqueeze(1).unsqueeze(1).repeat([1,self.num_nodes,self.num_nodes,1])

        # batch_size x from_nodes x to_nodes x (hidden_influence_dim + edge_feature_len + context_length)
        message_to_edge = torch.cat([all_influences,edges,context_repeated],dim=-1)
        assert(len(message_to_edge.size())==4)
        assert(message_to_edge.size()[1]==self.num_nodes)
        assert(message_to_edge.size()[2]==self.num_nodes)
        assert(message_to_edge.size()[3]==self.hidden_influence_dim+self.edge_feature_len+self.total_context_len)
        return message_to_edge


