import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean, scatter_softmax
from .utils import gather_nodes
import numpy as np
from .utils import gather_nodes, _dihedrals, _rbf, _orientations_coarse_gl
import math

############################ Graph Encoder ########################
###################################################################

def get_attend_mask(idx, mask):
    mask_attend = gather_nodes(mask.unsqueeze(-1), idx).squeeze(-1) # 一阶邻居节点的mask: 1代表节点存在, 0代表节点不存在
    mask_attend = mask.unsqueeze(-1) * mask_attend # 自身的mask*邻居节点的mask
    return mask_attend

class NeighborAttention(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, edge_drop=0):
        super(NeighborAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.edge_drop = edge_drop
        
        self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_K = nn.Linear(num_in, num_hidden, bias=False)
        self.W_V = nn.Linear(num_in, num_hidden, bias=False)
        self.Bias = nn.Sequential(
                                nn.Linear(num_hidden*3, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_heads)
                                )
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)

    def forward(self, h_V, h_E, center_id, batch_id):
        N = h_V.shape[0]
        E = h_E.shape[0]
        n_heads = self.num_heads
        d = int(self.num_hidden / n_heads)
        
        w = self.Bias(torch.cat([h_V[center_id], h_E],dim=-1)).view(E, n_heads, 1) 
        attend_logits = w/np.sqrt(d) 

        V = self.W_V(h_E).view(-1, n_heads, d) 
        attend = scatter_softmax(attend_logits, index=center_id, dim=0)
        h_V = scatter_sum(attend*V, center_id, dim=0).view([N, self.num_hidden])

        h_V_update = self.W_O(h_V)
        return h_V_update

class GNNModule(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, dropout=0):
        super(GNNModule, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(2)])
        self.attention = NeighborAttention(num_hidden, num_in, num_heads, edge_drop=0.0) # TODO: edge_drop
        self.dense = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden)
        )

    def forward(self, h_V, h_E, edge_idx, batch_id):
        center_id = edge_idx[0]
        dh = self.attention(h_V, h_E, center_id, batch_id)
        h_V = self.norm[0](h_V + self.dropout(dh))
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))
        return h_V
    
class StructureEncoder(nn.Module):
    def __init__(self,  hidden_dim, num_encoder_layers=3, dropout=0):
        """ Graph labeling network """
        super(StructureEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList([])
        for _ in range(num_encoder_layers):
            self.encoder_layers.append(nn.ModuleList([
                # Local_Module(hidden_dim, hidden_dim*2, is_attention=is_attention, dropout=dropout),
                GNNModule(hidden_dim, hidden_dim*2, dropout=dropout),
                GNNModule(hidden_dim, hidden_dim*2, dropout=dropout)
            ]))

    def forward(self, h_V, h_P, P_idx, batch_id):
        h_V = h_V
        # graph encoder
        for (layer1, layer2) in self.encoder_layers:
            h_EV_local = torch.cat([h_P, h_V[P_idx[1]]], dim=1)
            h_V = layer1(h_V, h_EV_local, P_idx, batch_id)
            
            h_EV_global = torch.cat([h_P, h_V[P_idx[1]]], dim=1)
            h_V = h_V + layer2(h_V, h_EV_global, P_idx, batch_id)
        return h_V



############################# Seq Decoder #########################
###################################################################
def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

class CNNDecoder(nn.Module):
    def __init__(self,hidden_dim, input_dim, vocab=20):
        super().__init__()
        
        # self.PosEnc = nn.Sequential(nn.Linear(20,hidden_dim),
        #                             nn.BatchNorm1d(hidden_dim),
        #                             nn.ReLU(),
        #                             nn.Linear(hidden_dim,hidden_dim),
        #                             nn.BatchNorm1d(hidden_dim),
        #                             nn.ReLU(),
        #                             nn.Linear(hidden_dim,hidden_dim))
        
        self.CNN = nn.Sequential(nn.Conv1d(input_dim, hidden_dim,5, padding=2),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.ReLU(),
                                   nn.Conv1d(hidden_dim, hidden_dim,5, padding=2),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.ReLU(),
                                   nn.Conv1d(hidden_dim, hidden_dim,5, padding=2))

        self.readout = nn.Linear(hidden_dim, vocab)
    
    def forward(self, h_V, batch_id):
        # pos = self.PosEnc(pos)
        # h_V = torch.cat([h_V,pos],dim=-1)
        h_V = h_V.unsqueeze(0).permute(0,2,1)
        hidden = self.CNN(h_V).permute(0,2,1).squeeze()
        logits = self.readout( hidden )
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, logits

class CNNDecoder2(nn.Module):
    def __init__(self,hidden_dim, input_dim, vocab=20):
        super().__init__()
        self.ConfNN = nn.Embedding(50, hidden_dim)
        
        # self.PosEnc = nn.Sequential(nn.Linear(20,hidden_dim),
        #                             nn.BatchNorm1d(hidden_dim),
        #                             nn.ReLU(),
        #                             nn.Linear(hidden_dim,hidden_dim),
        #                             nn.BatchNorm1d(hidden_dim),
        #                             nn.ReLU(),
        #                             nn.Linear(hidden_dim,hidden_dim))
        
        self.CNN = nn.Sequential(nn.Conv1d(hidden_dim+input_dim, hidden_dim,5, padding=2),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.ReLU(),
                                   nn.Conv1d(hidden_dim, hidden_dim,5, padding=2),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.ReLU(),
                                   nn.Conv1d(hidden_dim, hidden_dim,5, padding=2))

        self.readout = nn.Linear(hidden_dim, vocab)
    
    def forward(self, h_V, logits, batch_id):
        eps = 1e-5
        L = h_V.shape[0]
        idx = torch.argsort(-logits, dim=1)
        Conf = logits[range(L), idx[:,0]] / (logits[range(L), idx[:,1]] + eps)
        Conf = Conf.long()
        Conf = torch.clamp(Conf, 0, 49)
        h_C = self.ConfNN(Conf)
        
        # pos = self.PosEnc(pos)
        h_V = torch.cat([h_V,h_C],dim=-1)
        h_V = h_V.unsqueeze(0).permute(0,2,1)
        hidden = self.CNN(h_V).permute(0,2,1).squeeze()
        logits = self.readout( hidden )
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, logits


class ADesign(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim, 
        num_encoder_layers=3, num_decoder_layers=3, vocab=20, 
        k_neighbors=30, dropout=0.1, **kwargs):
        """ Graph labeling network """
        super(ADesign, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.top_k = k_neighbors
        self.num_rbf = 16
        self.num_positional_embeddings = 16

        node_in, edge_in = 12, 16+7
        self.node_embedding = nn.Linear(node_in, node_features, bias=True)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=True)
        self.norm_nodes = nn.BatchNorm1d(node_features)
        self.norm_edges = nn.BatchNorm1d(edge_features)

        self.W_v = nn.Sequential(
            nn.Linear(node_features, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=True)
        )
        
        
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True) 
        self.W_f = nn.Linear(edge_features, hidden_dim, bias=True)

        self.encoder = StructureEncoder(hidden_dim, num_decoder_layers, dropout)

        self.decoder = CNNDecoder(hidden_dim, hidden_dim)
        self.decoder2 = CNNDecoder2(hidden_dim, hidden_dim)
        self._init_params()
    
    def forward(self, h_V, h_P, P_idx, batch_id):
        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_P = self.W_e(self.norm_edges(self.edge_embedding(h_P)))
        
        h_V = self.encoder(h_V, h_P, P_idx, batch_id)
        log_probs0, logits = self.decoder(h_V, batch_id)
        
        log_probs, logits = self.decoder2(h_V, logits, batch_id)
        return log_probs, log_probs0
        
    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _full_dist(self, X, mask, top_k=30, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = (1. - mask_2D)*10000 + mask_2D* torch.sqrt(torch.sum(dX**2, 3) + eps)

        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * (D_max+1)
        D_neighbors, E_idx = torch.topk(D_adjust, min(top_k, D_adjust.shape[-1]), dim=-1, largest=False)
        return D_neighbors, E_idx  

    def _get_features(self, S, score, X, mask):
        mask_bool = (mask==1)
        
        B, N, _,_ = X.shape
        X_ca = X[:,:,1,:]
        D_neighbors, E_idx = self._full_dist(X_ca, mask, 30) # TODO: change_k

        # sequence
        S = torch.masked_select(S, mask_bool)
        if score is not None:
            score = torch.masked_select(score, mask_bool)


        # node feature
        _V = _dihedrals(X) 
        _V = torch.masked_select(_V, mask_bool.unsqueeze(-1)).reshape(-1,_V.shape[-1])

        # edge feature
        _E = torch.cat((_rbf(D_neighbors, self.num_rbf), _orientations_coarse_gl(X, E_idx)), -1) # [4,387,387,23]
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1) # 一阶邻居节点的mask: 1代表节点存在, 0代表节点不存在
        mask_attend = (mask.unsqueeze(-1) * mask_attend) == 1 # 自身的mask*邻居节点的mask
        _E = torch.masked_select(_E, mask_attend.unsqueeze(-1)).reshape(-1,_E.shape[-1])

        
        # edge index
        shift = mask.sum(dim=1).cumsum(dim=0) - mask.sum(dim=1)
        src = shift.view(B,1,1) + E_idx
        src = torch.masked_select(src, mask_attend).view(1,-1)
        dst = shift.view(B,1,1) + torch.arange(0, N, device=src.device).view(1,-1,1).expand_as(mask_attend)
        dst = torch.masked_select(dst, mask_attend).view(1,-1)
        E_idx = torch.cat((dst, src), dim=0).long()
        
        # 3D point
        sparse_idx = mask.nonzero()
        X = X[sparse_idx[:,0],sparse_idx[:,1],:,:]
        batch_id = sparse_idx[:,0]

        return X, S, score, _V, _E, E_idx, batch_id