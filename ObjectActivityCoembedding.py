from argparse import ArgumentError
from importlib.util import module_for_loader
import os
from platform import node
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
from GraphTranslatorModule import GraphTranslatorModule


class ObjectActivityCoembeddingModule(LightningModule):
    def __init__(self, model_configs):
        
        super().__init__()

        self.cfg = model_configs
        self.embedding_size = self.cfg.c_len


        ## Object Transition Encoder
        self.graph_cnn = geom_nn.Sequential('x, edge_index',
                                            [(geom_nn.GCNConv(self.cfg.n_len, self.embedding_size*4), 'x, edge_index -> x'),
                                            nn.ReLU(inplace=True),
                                            (geom_nn.GCNConv(self.embedding_size*4, self.embedding_size*2), 'x, edge_index -> x'),
                                            ])
        self.context_from_graph_encodings = nn.Sequential(nn.Linear(self.embedding_size*2, self.embedding_size*2),
                                                          nn.ReLU(),
                                                          nn.Linear(self.embedding_size*2, self.embedding_size)
                                                          )

        self.obj_seq_encoder_transformer_layer = torch.nn.TransformerEncoderLayer(self.embedding_size, nhead=2)
        self.obj_seq_encoder_transformer = torch.nn.TransformerEncoder(self.obj_seq_encoder_transformer_layer, num_layers=4)

        # graph_module = GraphTranslatorModule(model_configs=model_configs)
        # self.time_context = graph_module.mlp_context
        # self.obj_seq_decoder = graph_module
        # self.obj_graph_loss = lambda x,y: (nn.CrossEntropyLoss(reduction='mean')(x.squeeze(-1).permute(0,2,1), y.squeeze(-1)))

        ## Activity Encoder : LM + MLP
        self.embed_context_activity = nn.Linear(self.cfg.n_activities, self.embedding_size)

        self.activity_seq_encoder_transformer_layer = torch.nn.TransformerEncoderLayer(self.embedding_size, nhead=2)
        self.activity_seq_encoder_transformer = torch.nn.TransformerEncoder(self.activity_seq_encoder_transformer_layer, num_layers=4)

        self.activity_decoder_mlp = nn.Sequential(nn.Linear(self.embedding_size, self.embedding_size),
                                                  nn.ReLU(),
                                                  nn.Linear(self.embedding_size, self.cfg.n_activities)
                                                  )

        self.activity_prediction_loss = lambda x,y: (nn.CrossEntropyLoss(reduction='mean')(x.squeeze(-1).permute(0,2,1), y.squeeze(-1).long()))



    def context_loss(self, xc, yc):
        batch_size = xc.size()[0]
        self.xgrid, self.ygrid = torch.meshgrid(torch.arange(batch_size), torch.arange(batch_size), indexing='ij')
        same = (((self.xgrid==self.ygrid).to(int).reshape(-1)*2)-1).to('cuda')
        return torch.nn.CosineEmbeddingLoss(reduction='mean')(xc[self.xgrid.reshape(-1),:], yc[self.ygrid.reshape(-1),:], same)

    def graph_encoder(self, nodes, edges):
        """
        Args:
            nodes: batch_size x sequence_length+1 x num_nodes x node_feature_len
            edges: batch_size x sequence_length+1 x num_nodes x num_nodes x edge_feature_len
        Returns:
            latents: batch_size x sequence_length x embedding_size
        """
        batch_size = nodes.size()[0]
        sequence_len_plus_one = nodes.size()[1]
        mat2idx = lambda e_mat: (torch.argwhere(e_mat == 1)).transpose(1,0)
        batch = torch.arange(batch_size*sequence_len_plus_one).repeat(self.cfg.n_nodes,1).transpose(1,0).reshape(-1).to(int).to('cuda')
        spatial_edges = torch.cat([mat2idx(edges[b,s,:,:])+(b*(sequence_len_plus_one*self.cfg.n_nodes)+s*self.cfg.n_nodes) for s in range(sequence_len_plus_one) for b in range(batch_size)], dim=-1)
        assert spatial_edges.size()[0] == 2
        temporal_edges = torch.tensor([[i,self.cfg.n_nodes+i] for i in range((batch_size*sequence_len_plus_one-1)*self.cfg.n_nodes) if i%sequence_len_plus_one!=0], device='cuda').permute(1,0)
        assert temporal_edges.size()[0] == 2
        all_edges = torch.cat([spatial_edges, temporal_edges], dim=-1)
        graphs_in = geom_nn.global_mean_pool(self.graph_cnn(nodes.reshape(batch_size*sequence_len_plus_one*self.cfg.n_nodes, self.cfg.n_len), all_edges), batch=batch)
        latent_per_graph = self.context_from_graph_encodings(graphs_in)
        assert latent_per_graph.size()[0] == batch_size*sequence_len_plus_one
        latent_per_graph = latent_per_graph.reshape(batch_size, sequence_len_plus_one, self.embedding_size)
        latent = latent_per_graph[:,1:,:] - latent_per_graph[:,:-1,:]
        assert latent.size()[0] == batch_size
        assert latent.size()[1] == sequence_len_plus_one - 1
        assert latent.size()[2] == self.embedding_size
        return latent

    def activity_encoder(self, activity):
        """
        Args:
            activity: batch_size x sequence_length
        Return:
            _: batch_size x sequence_length x n_activities
        """
        return self.embed_context_activity(F.one_hot(activity.to(int), num_classes=self.cfg.n_activities).to(torch.float32))

    def activity_decoder(self, latent_vector, ground_truth=None):
        """
        Args:
            latent_vector: batch_size x sequence_length x embedding_size
            ground_truth: batch_size x sequence_length
        Return:
            output_activity: batch_size x sequence_length x n_activities
            activity_pred_loss: batch_size x sequence_length
        """

        # activity_embeddings = self.embed_context_activity.weight
        # assert activity_embeddings.size()[0] == self.c_len
        # assert activity_embeddings.size()[1] == self.cfg.n_activities
        # output_activity = torch.nn.CosineSimilarity(dim=1)(activity_embeddings.unsqueeze(0), latent_vector.unsqueeze(-1))
        
        output_activity = self.activity_decoder_mlp(latent_vector)
        
        if ground_truth is not None:
            activity_pred_loss = self.activity_prediction_loss(output_activity, ground_truth)
        else:
            activity_pred_loss = None
        output_activity = F.softmax(output_activity, dim=-1)
        return output_activity, activity_pred_loss

    def seq_encoder(self, latents, time_context, seq_type):
        """
        Args:
            latents: batch_size x sequence_length x embedding_size
        Return:
            encoded: batch_size x sequence_length x embedding_size
        """
        if seq_type == 'object':
            transformer = self.obj_seq_encoder_transformer
        elif seq_type == 'activity':
            transformer = self.activity_seq_encoder_transformer
        else:
            raise ArgumentError(f'Sequence type must be object or activity, not {seq_type}')

        batch_size, sequence_length, _ = latents.size()
        mask = torch.zeros((sequence_length*batch_size, sequence_length*batch_size), device='cuda')
        for b in range(batch_size):
            mask[b*sequence_length:(b+1)*sequence_length, b*sequence_length:(b+1)*sequence_length] = 1
        # mask = torch.block_diag(tuple([torch.ones(sequence_length, sequence_length) for _ in range(batch_size)]))
        mask =  mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        encoded = transformer(latents.reshape(batch_size * sequence_length, self.embedding_size) + time_context.reshape(batch_size * sequence_length, self.embedding_size), mask=mask).reshape(batch_size, sequence_length, self.embedding_size)
        return encoded

    def latent_similarity_loss(self, latent_obj, latent_act):
        """
        Args:
            latent_obj: batch_size x sequence_len x embedding_size
            latent_act: batch_size x sequence_len x embedding_size
        """
        latent_obj = latent_obj.reshape(-1, self.embedding_size)
        latent_act = latent_act.reshape(-1, self.embedding_size)
        flat_batch_size = latent_obj.size()[0]
        xgrid, ygrid = torch.meshgrid(torch.arange(flat_batch_size), torch.arange(flat_batch_size), indexing='ij')
        # xgrid = torch.cat([torch.diagonal(xgrid, offset=offset) for offset in range(self.num_negative_diagonals)])
        # ygrid = torch.cat([torch.diagonal(ygrid, offset=offset) for offset in range(self.num_negative_diagonals)])
        same = (((xgrid==ygrid).to(int).reshape(-1)*2)-1).to('cuda')
        return torch.nn.CosineEmbeddingLoss(reduction='mean')(latent_obj[xgrid.reshape(-1),:], latent_act[ygrid.reshape(-1),:], same)



    def forward(self, graph_seq_nodes, graph_seq_edges, activity_seq, time_context, graph_seq_dyn_edges=None):
        """
        Args:
            graph_seq_nodes: batch_size x sequence_length+1 x num_nodes x node_feature_len
            graph_seq_edges: batch_size x sequence_length+1 x num_nodes x num_nodes x edge_feature_len
            activity_seq: batch_size x sequence_length
            activity_seq: batch_size x sequence_length x context_length
        """

        activity_seq = activity_seq[:,:-1]
        time_context = time_context[:,:-1,:]

        batch_size, sequence_len_plus_one, num_nodes, node_feature_len = graph_seq_nodes.size()
        batch_size_e, sequence_len_plus_one_e, num_f_nodes, num_t_nodes = graph_seq_edges.size()
        batch_size_act, sequence_len = activity_seq.size()
        
        # Sanity check input dimensions
        assert batch_size == batch_size_e, "Different edge and node batch sizes"
        assert batch_size == batch_size_act, "Different edge and node batch sizes"
        assert sequence_len_plus_one == sequence_len_plus_one_e, "Different edge and node sequence lengths"
        assert sequence_len_plus_one == sequence_len + 1, "Different graph and activity sequence lengths"
        assert self.cfg.n_len == node_feature_len, (str(self.cfg.n_len) +'!='+ str(node_feature_len))
        assert self.cfg.n_nodes == num_nodes, (str(self.cfg.n_nodes) +'!='+ str(num_nodes))
        assert self.cfg.n_nodes == num_f_nodes, (str(self.cfg.n_nodes) +'!='+ str(num_f_nodes))
        assert self.cfg.n_nodes == num_t_nodes, (str(self.cfg.n_nodes) +'!='+ str(num_t_nodes))

        graph_latents = self.graph_encoder(graph_seq_nodes, graph_seq_edges)
        graph_latents = self.seq_encoder(graph_latents, time_context, seq_type='object')

        activity_latents = self.activity_encoder(activity_seq)
        activity_latents = self.seq_encoder(activity_latents, time_context, seq_type='activity')

        latent_loss = self.latent_similarity_loss(graph_latents, activity_latents)
        
        # if time_context is not None:
        #     time_latents = self.time_context(time_context)
        #     latent_loss += self.latent_similarity_loss(graph_latents, time_latents)
        #     context = F.normalize(graph_latents + activity_latents + time_latents)
        # else:
        #     context = F.normalize(graph_latents + activity_latents)

        # input_nodes = graph_seq_nodes[:,:-1,:,:]
        # input_edges = graph_seq_edges[:,:-1,:,:]
        # output_edges = graph_seq_edges[:,1:,:,:]
        # edges_pred, _, _, _ = self.obj_seq_decoder(input_edges.reshape(batch_size*sequence_len, self.cfg.n_nodes, self.cfg.n_nodes), 
        #                                            input_nodes.reshape(batch_size*sequence_len, self.cfg.n_nodes, self.cfg.n_len), 
        #                                            graph_latents.reshape(batch_size*sequence_len, self.embedding_size))
        # # dyn_edges_flat = graph_seq_dyn_edges.reshape(batch_size*sequence_len, self.cfg.n_nodes, self.cfg.n_nodes, 1)
        # # edges_pred[dyn_edges_flat == 0] = -float('inf')
        # graph_loss = self.obj_graph_loss(edges_pred, output_edges.reshape(batch_size*sequence_len, self.cfg.n_nodes, self.cfg.n_nodes))
        # edges_inferred = F.softmax(edges_pred, dim=-1)
        # # edges_inferred[dyn_edges_flat == 0] = input_edges[dyn_edges_flat == 0]
        # edges_inferred = edges_inferred.reshape(batch_size, sequence_len, self.cfg.n_nodes, self.cfg.n_nodes)

        output_activity, activity_pred_loss = self.activity_decoder(activity_latents, ground_truth=activity_seq)

        results = {
            'output' : {
                        # 'object': edges_inferred,
                        'activity': output_activity},
            'loss' : {
                    #   'object': graph_loss,
                      'activity': activity_pred_loss,
                      'similarity': latent_loss},
            'details' : {}
        }

        return results

        
    def training_step(self, batch, batch_idx):
        results = self(batch['nodes'], batch['edges'], batch['activity'], batch['context_time'])
        self.log('Train loss',results['loss'])
        # self.log('Train',results['details'])
        res = results['loss']['similarity']
        return res

    def test_step(self, batch, batch_idx):
        results = self(batch['nodes'], batch['edges'], batch['activity'], batch['context_time'])
        self.log('Test loss',results['loss'])
        # self.log('Test',results['details'])
        return 

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

