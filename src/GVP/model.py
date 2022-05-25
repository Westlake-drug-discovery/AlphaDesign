import numpy as np
import torch
import torch.nn as nn
from .gvp import GVP, GVPConvLayer, LayerNorm, tuple_index
from torch.distributions import Categorical
from torch_scatter import scatter_mean


class CPDModel(torch.nn.Module):
    '''
    GVP-GNN for structure-conditioned autoregressive 
    protein design as described in manuscript.
    
    Takes in protein structure graphs of type `torch_geometric.data.Data` 
    or `torch_geometric.data.Batch` and returns a categorical distribution
    over 20 amino acids at each position in a `torch.Tensor` of 
    shape [n_nodes, 20].
    
    Should be used with `gvp.data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.
    
    The standard forward pass requires sequence information as input
    and should be used for training or evaluating likelihood.
    For sampling or design, use `self.sample`.
    
    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param node_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :param num_layers: number of GVP-GNN layers in each of the encoder
                       and decoder modules
    :param drop_rate: rate to use in all dropout layers
    '''
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,
                 num_layers=3, drop_rate=0.1):
    
        super(CPDModel, self).__init__()
        
        self.W_v = nn.Sequential(
            GVP(node_in_dim, node_h_dim, activations=(None, None), use_norm=False),
            LayerNorm(node_h_dim)
        )
        self.W_e = nn.Sequential(
            GVP(edge_in_dim, edge_h_dim, activations=(None, None), use_norm=True),
            LayerNorm(edge_h_dim)
        )
        
        self.encoder_layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate, use_norm=False) 
            for _ in range(num_layers))
        
        self.W_s = nn.Embedding(20, 20)
        edge_h_dim = (edge_h_dim[0] + 20, edge_h_dim[1])
      
        self.decoder_layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, 
                             drop_rate=drop_rate, autoregressive=True, use_norm=False) 
            for _ in range(num_layers))
        
        self.W_out = GVP(node_h_dim, (20, 0), activations=(None, None), use_norm=False)

    def forward(self, h_V, edge_index, h_E, seq):
        '''
        Forward pass to be used at train-time, or evaluating likelihood.
        
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: int `torch.Tensor` of shape [num_nodes]
        '''
        h_V = self.W_v(h_V) # [1963, 100], [1963, 16, 3]
        h_E = self.W_e(h_E) # [58890, 32], [58890, 1, 3]
        
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)
        
        encoder_embeddings = h_V
        
        h_S = self.W_s(seq)
        h_S = h_S[edge_index[0]]
        h_S[edge_index[0] >= edge_index[1]] = 0
        h_E = (torch.cat([h_E[0], h_S], dim=-1), h_E[1])
        
        for layer in self.decoder_layers:
            h_V = layer(h_V, edge_index, h_E, autoregressive_x = encoder_embeddings)
        
        logits = self.W_out(h_V)
        
        return logits
    
    def sample(self, h_V, edge_index, h_E, n_samples, temperature=0.1):
        '''
        Samples sequences autoregressively from the distribution
        learned by the model.
        
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param n_samples: number of samples
        :param temperature: temperature to use in softmax 
                            over the categorical distribution
        
        :return: int `torch.Tensor` of shape [n_samples, n_nodes] based on the
                 residue-to-int mapping of the original training data
        '''
        
        with torch.no_grad():
        
            device = edge_index.device
            L = h_V[0].shape[0]
            
            h_V = self.W_v(h_V) # [n, 100], [n, 16, 3]
            h_E = self.W_e(h_E) # [e, 32], [e, 1, 3]
            
            for layer in self.encoder_layers:
                h_V = layer(h_V, edge_index, h_E)   
            
            h_V = (h_V[0].repeat(n_samples, 1),
                   h_V[1].repeat(n_samples, 1, 1)) # [n*100, 100],  [n*100, 16, 3]
            
            h_E = (h_E[0].repeat(n_samples, 1),
                   h_E[1].repeat(n_samples, 1, 1)) # [e*100, 32], [e*100, 1, 3]
            
            edge_index = edge_index.expand(n_samples, -1, -1) # [2, e] --> [100, 2, e]
            offset = L * torch.arange(n_samples, device=device).view(-1, 1, 1)
            edge_index = torch.cat(tuple(edge_index + offset), dim=-1)
            
            seq = torch.zeros(n_samples * L, device=device, dtype=torch.int)
            h_S = torch.zeros(n_samples * L, 20, device=device)
    
            h_V_cache = [(h_V[0].clone(), h_V[1].clone()) for _ in self.decoder_layers]
            
            for i in range(L):
                
                h_S_ = h_S[edge_index[0]]
                h_S_[edge_index[0] >= edge_index[1]] = 0
                h_E_ = (torch.cat([h_E[0], h_S_], dim=-1), h_E[1])
                        
                edge_mask = edge_index[1] % L == i
                edge_index_ = edge_index[:, edge_mask]
                h_E_ = tuple_index(h_E_, edge_mask)
                node_mask = torch.zeros(n_samples * L, device=device, dtype=torch.bool) # [n*100]
                node_mask[i::L] = True
                
                for j, layer in enumerate(self.decoder_layers):
                    # known: h_V_cache[j]
                    out = layer(h_V_cache[j], edge_index_, h_E_,
                               autoregressive_x=h_V_cache[0], node_mask=node_mask) # [n*100, 20]
                    
                    out = tuple_index(out, node_mask) # [100, 20]
                    
                    if j < len(self.decoder_layers)-1:
                        h_V_cache[j+1][0][i::L] = out[0]
                        h_V_cache[j+1][1][i::L] = out[1]
                
                logits = self.W_out(out) # [100, 20]
                seq[i::L] = Categorical(logits=logits / temperature).sample()
                h_S[i::L] = self.W_s(seq[i::L])
                
            return seq.view(n_samples, L)
  
    def test_recovery(self, protein):
        h_V = (protein.node_s, protein.node_v)
        h_E = (protein.edge_s, protein.edge_v) 
        sample = self.sample(h_V, protein.edge_index, h_E, n_samples=1)
        return sample