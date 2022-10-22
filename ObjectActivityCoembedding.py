from argparse import ArgumentError
import os
from copy import deepcopy
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
        # self.train_prediction = False

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
        self.obj_seq_encoder_transformer = torch.nn.TransformerEncoder(self.obj_seq_encoder_transformer_layer, num_layers=1)

        graph_module = GraphTranslatorModule(model_configs=model_configs)
        self.time_context = graph_module.mlp_context
        self.obj_seq_decoder = graph_module
        self.obj_graph_loss = lambda x,y: (nn.CrossEntropyLoss(reduction='mean')(x.permute(0,2,1), y.argmax(-1)))

        ## Activity Encoder
        self.embed_context_activity = nn.Linear(self.cfg.n_activities, self.embedding_size)

        self.activity_seq_encoder_transformer_layer = torch.nn.TransformerEncoderLayer(self.embedding_size, nhead=2)
        self.activity_seq_encoder_transformer = torch.nn.TransformerEncoder(self.activity_seq_encoder_transformer_layer, num_layers=1)

        self.activity_decoder_mlp = nn.Sequential(nn.Linear(self.embedding_size, self.embedding_size),
                                                  nn.ReLU(),
                                                  nn.Linear(self.embedding_size, self.cfg.n_activities)
                                                  )

        self.activity_prediction_loss = lambda x,y: (nn.CrossEntropyLoss(reduction='mean')(x.squeeze(-1).permute(0,2,1), y.squeeze(-1).long()))

        ## Prediction Model
        self.prediction_transformer_layer = torch.nn.TransformerEncoderLayer(self.embedding_size, nhead=2)
        self.prediction_transformer = torch.nn.TransformerEncoder(self.prediction_transformer_layer, num_layers=1)
        ### TODO Maithili : Change to cosine embedding loss
        self.predictive_latent_loss = torch.nn.MSELoss()   #(reduction='mean')
        

    def context_loss(self, xc, yc):
        batch_size = xc.size()[0]
        self.xgrid, self.ygrid = torch.meshgrid(torch.arange(batch_size), torch.arange(batch_size), indexing='ij')
        same = (((self.xgrid==self.ygrid).to(int).view(-1)*2)-1).to('cuda')
        return torch.nn.CosineEmbeddingLoss(reduction='mean')(xc[self.xgrid.view(-1),:], yc[self.ygrid.view(-1),:], same)

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
        graphs_in = geom_nn.global_mean_pool(self.graph_cnn(nodes.view(batch_size*sequence_len_plus_one*self.cfg.n_nodes, self.cfg.n_len), all_edges), batch=batch)
        latent_per_graph = self.context_from_graph_encodings(graphs_in)
        assert latent_per_graph.size()[0] == batch_size*sequence_len_plus_one
        latent_per_graph = latent_per_graph.view(batch_size, sequence_len_plus_one, self.embedding_size)
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
        return self.embed_context_activity(activity)

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
        
        activity_pred_loss = None
        if ground_truth is not None:
            activity_pred_loss = self.activity_prediction_loss(output_activity, ground_truth)

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
        elif seq_type == 'predictive':
            transformer = self.prediction_transformer
        else:
            raise ArgumentError(f'Sequence type must be object or activity, not {seq_type}')

        batch_size, sequence_length, _ = latents.size()
        mask = torch.zeros((sequence_length*batch_size, sequence_length*batch_size), device='cuda')
        for b in range(batch_size):
            mask[b*sequence_length:(b+1)*sequence_length, b*sequence_length:(b+1)*sequence_length] = 1
        if seq_type == 'predictive':
            mask *= torch.ones(sequence_length*batch_size, sequence_length*batch_size).to('cuda').tril()
        # mask = torch.block_diag(tuple([torch.ones(sequence_length, sequence_length) for _ in range(batch_size)]))
        mask =  mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        encoded = transformer(latents.view(batch_size * sequence_length, self.embedding_size) + time_context.view(batch_size * sequence_length, self.embedding_size), mask=mask).view(batch_size, sequence_length, self.embedding_size)
        return encoded

    def latent_similarity_loss(self, latent_obj, latent_act):
        """
        Args:
            latent_obj: batch_size x sequence_len x embedding_size
            latent_act: batch_size x sequence_len x embedding_size
        """
        latent_obj = latent_obj.view(-1, self.embedding_size)
        latent_act = latent_act.view(-1, self.embedding_size)
        flat_batch_size = latent_obj.size()[0]
        xgrid, ygrid = torch.meshgrid(torch.arange(flat_batch_size), torch.arange(flat_batch_size), indexing='ij')
        # xgrid = torch.cat([torch.diagonal(xgrid, offset=offset) for offset in range(self.num_negative_diagonals)])
        # ygrid = torch.cat([torch.diagonal(ygrid, offset=offset) for offset in range(self.num_negative_diagonals)])
        same = (((xgrid==ygrid).to(int).view(-1)*2)-1).to('cuda')

        ### TODO Maithili : Add 'M' parameter
        return torch.nn.CosineEmbeddingLoss(reduction='mean')(latent_obj[xgrid.reshape(-1),:], latent_act[ygrid.reshape(-1),:], same)

    def encode_graph(self, graph_seq_nodes, graph_seq_edges):
        graph_latents = F.normalize(self.graph_encoder(graph_seq_nodes, graph_seq_edges), dim=-1)
        # graph_latents = self.seq_encoder(graph_latents, time_context, seq_type='object')
        input_nodes = graph_seq_nodes[:,:-1,:,:]
        input_edges = graph_seq_edges[:,:-1,:,:]
        output_edges = graph_seq_edges[:,1:,:,:]
        batch_size, sequence_len, _, _ = input_nodes.size()

        edges_inferred, _, _ = self.obj_seq_decoder(input_edges.view(batch_size*sequence_len, self.cfg.n_nodes, self.cfg.n_nodes), 
                                                   input_nodes.view(batch_size*sequence_len, self.cfg.n_nodes, self.cfg.n_len), 
                                                   graph_latents.view(batch_size*sequence_len, self.embedding_size))
        # dyn_edges_flat = graph_seq_dyn_edges.view(batch_size*sequence_len, self.cfg.n_nodes, self.cfg.n_nodes, 1)
        # edges_inferred[dyn_edges_flat == 0] = -float('inf')
        graph_autoenc_loss = self.obj_graph_loss(edges_inferred, output_edges.view(batch_size*sequence_len, self.cfg.n_nodes, self.cfg.n_nodes))

        return graph_latents, graph_autoenc_loss

    def encode_activity(self, activity_seq):
        activity_latents = F.normalize(self.activity_encoder(F.one_hot(activity_seq.to(int), num_classes=self.cfg.n_activities).to(torch.float32)), dim=-1)
        # activity_latents = self.seq_encoder(activity_latents, time_context, seq_type='activity')
        _, activity_autoenc_loss = self.activity_decoder(activity_latents, ground_truth=activity_seq)
        return activity_latents, activity_autoenc_loss

    def predict(self, latents, time_context, latents_expected=None):
        batch_size, sequence_len, _ = latents.size()
        
        pred_latents = F.normalize(self.seq_encoder(latents, time_context, seq_type='predictive'), dim=-1)

        latent_predictive_loss = None
        if latents_expected is not None:
            latent_predictive_loss = self.predictive_latent_loss(pred_latents, latents_expected)    # , target=torch.Tensor([1]).to('cuda'))
        
        return pred_latents, latent_predictive_loss

    def decode(self, latents, input_nodes, input_edges, output_edges=None, output_activity=None):
        batch_size, sequence_len, _, _ = input_edges.size()

        pred_edges, _, _ = self.obj_seq_decoder(input_edges.view(batch_size*(sequence_len), self.cfg.n_nodes, self.cfg.n_nodes), 
                                                   input_nodes.view(batch_size*(sequence_len), self.cfg.n_nodes, self.cfg.n_len), 
                                                   latents.view(batch_size*(sequence_len), self.embedding_size))

        # pred_edges = input_edges.view(batch_size*(sequence_len), self.cfg.n_nodes, self.cfg.n_nodes)


        graph_pred_loss = None
        if output_edges is not None:
            graph_pred_loss = self.obj_graph_loss(pred_edges, output_edges.view(batch_size*(sequence_len), self.cfg.n_nodes, self.cfg.n_nodes))
        pred_edges = F.softmax(pred_edges, dim=-1).view(batch_size, sequence_len, self.cfg.n_nodes, self.cfg.n_nodes)
        
        pred_activity, activity_pred_loss = self.activity_decoder(latents, ground_truth=output_activity)

        return pred_edges, pred_activity, graph_pred_loss, activity_pred_loss


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

        graph_latents, graph_autoenc_loss = self.encode_graph(graph_seq_nodes, graph_seq_edges)

        activity_latents, activity_autoenc_loss = self.encode_activity(activity_seq)
       
        latent_similarity_loss = self.latent_similarity_loss(graph_latents, activity_latents)
        
        # if time_context is not None:
        #     time_latents = self.time_context(time_context)
        #     latent_similarity_loss += self.latent_similarity_loss(graph_latents, time_latents)
        #     context = F.normalize(graph_latents + activity_latents + time_latents, dim=-1)
        # else:

        keep_activity_mask = (torch.rand((activity_latents.size()[0], activity_latents.size()[1])) > self.cfg.activity_dropout_prob).unsqueeze(-1).to('cuda')
        latents = F.normalize(graph_latents + (activity_latents * keep_activity_mask), dim=-1)

        ## Prediction
        # if self.train_prediction:
        input_nodes_forward = graph_seq_nodes[:,1:-1,:,:]
        input_edges_forward = graph_seq_edges[:,1:-1,:,:]
        output_edges_forward = graph_seq_edges[:,2:,:,:]

        pred_latents, latent_predictive_loss = self.predict(latents[:,:-1], 
                                                            time_context[:,:-1], 
                                                            latents_expected=latents[:,1:])

        pred_edges, pred_activity, graph_pred_loss, activity_pred_loss = self.decode(latents=pred_latents, 
                                                                                    input_nodes=input_nodes_forward,
                                                                                    input_edges=input_edges_forward,
                                                                                    output_edges=output_edges_forward,
                                                                                    output_activity=activity_seq[:,1:])

        # print(output_edges_forward.size())    
        # print(pred_edges.size())    
        # print(activity_seq.size())    
        # print(pred_activity.size())    

        accuracy_object = (output_edges_forward.argmax(-1) == pred_edges.argmax(-1)).sum()/torch.numel(output_edges_forward.argmax(-1))
        accuracy_activity = (activity_seq[:,1:] == pred_activity.argmax(-1)).sum()/torch.numel(activity_seq[:,1:])

        # else:
        #     graph_pred_loss = 0
        #     activity_pred_loss = 0
        #     latent_predictive_loss = 0
        #     pred_edges = None
        #     pred_activity = None

        results = {
            'output' : {
                        'object': pred_edges,
                        'activity': pred_activity
                        },
            'loss' : {
                      'object_autoencoder': graph_autoenc_loss,
                      'object_pred': graph_pred_loss,
                      'activity_autoencoder': activity_autoenc_loss,
                      'activity_pred': activity_pred_loss,
                      'latent_similarity': latent_similarity_loss,
                      'latent_pred': latent_predictive_loss
                      },
            'accuracies' : {
                        'object': accuracy_object,
                        'activity': accuracy_activity
            }
        }

        return results


    def evaluate_prediction(self, batch, num_steps=1):

        graph_seq_nodes = batch.get('nodes')
        graph_seq_edges = batch.get('edges')
        activity_seq = batch.get('activity')[:,:-1]
        time_context = batch.get('context_time')[:,:-1,:]

        # if len(activity_seq.size()) == 2:
        #     activity_seq = F.one_hot(activity_seq.to(int), num_classes=self.cfg.n_activities).to(torch.float32)

        self.obj_seq_decoder.evaluate = True

        graph_latents, _ = self.encode_graph(graph_seq_nodes, graph_seq_edges)
        latents = graph_latents

        if activity_seq is not None:
            activity_latents, _ = self.encode_activity(activity_seq)
            keep_activity_mask = (torch.rand((activity_latents.size()[0], activity_latents.size()[1])) > self.cfg.activity_dropout_prob).unsqueeze(-1).to('cuda')
            latents = graph_latents + (activity_latents * keep_activity_mask)

        latents = F.normalize(latents, dim=-1)

        ## Prediction
        routine_len = latents.size()[1] - num_steps
        input_nodes_forward = graph_seq_nodes[:,1:1+routine_len,:,:]
        input_edges_forward = graph_seq_edges[:,1:1+routine_len,:,:]
        latents_forward = latents[:,:routine_len]
        results = {'object':[], 'activity':[]} #, 'gt_object':[], 'gt_activity':[]}


        for step in range(num_steps):
            latents_forward, _ = self.predict(latents_forward, time_context[:,step:step+routine_len])
            pred_edges, pred_activity, _, _ = self.decode(latents=latents_forward, input_nodes=input_nodes_forward, input_edges=input_edges_forward)

            # input_nodes_forward = input_nodes_forward[:,1:,:,:]
            input_edges_forward = pred_edges

            results['object'].append(pred_edges)
            # results['gt_object'].append(graph_seq_edges[:,2+step:,:,:])
            results['activity'].append(pred_activity)
            # results['gt_activity'].append(activity_seq[:,1+step:])

        return results

        
    def training_step(self, batch, batch_idx):
        results = self(batch['nodes'], batch['edges'], batch['activity'], batch['context_time'])
        self.log('Train loss',results['loss'])
        self.log('Train accuracy',results['accuracies'])
        # self.log('Train',results['details'])
        # res = sum(results['loss'].values())
        res = 0
        res += results['loss']['object_autoencoder']
        # res += results['loss']['object_pred']
        res += results['loss']['activity_autoencoder']
        # res += results['loss']['activity_pred']
        res += results['loss']['latent_similarity']
        res += results['loss']['latent_pred']
        return res

    def test_step(self, batch, batch_idx):
        results = self(batch['nodes'], batch['edges'], batch['activity'], batch['context_time'])
        self.log('Test loss',results['loss'])
        # self.log('Test',results['details'])
        return 

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

