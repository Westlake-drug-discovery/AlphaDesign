import torch
import torch.nn as nn
from .struct2seq import struct2seq
import numpy as np
from .struct2seq.self_attention import *

# def featurize(batch, device, shuffle_fraction=0.):
#     """ Pack and pad batch into torch tensors """
#     alphabet = 'ACDEFGHIKLMNPQRSTVWY'
#     B = len(batch)
#     lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
#     L_max = max([len(b['seq']) for b in batch])
#     X = np.zeros([B, L_max, 4, 3])
#     S = np.zeros([B, L_max], dtype=np.int32)

#     def shuffle_subset(n, p):
#         n_shuffle = np.random.binomial(n, p)
#         ix = np.arange(n)
#         ix_subset = np.random.choice(ix, size=n_shuffle, replace=False)
#         ix_subset_shuffled = np.copy(ix_subset)
#         np.random.shuffle(ix_subset_shuffled)
#         ix[ix_subset] = ix_subset_shuffled
#         return ix


class NIPS19_model(nn.Module):
    def __init__(self, args, use_mpnn=False):
        super(NIPS19_model, self).__init__()
        
        self.device = 'cuda:0'
        self.smoothing = args.smoothing
        
        self.model = struct2seq.Struct2Seq(
            num_letters=args.vocab_size, 
            node_features=args.hidden,
            edge_features=args.hidden, 
            hidden_dim=args.hidden,
            k_neighbors=args.k_neighbors,
            protein_features=args.features,
            dropout=args.dropout,
            use_mpnn=use_mpnn
        )
        
    def forward(self, X, S, lengths, mask):
        log_probs = self.model(X, S, lengths, mask)
        return log_probs

    def sample(self, X, L, mask=None, temperature=1.0):
        """ Autoregressive decoding of a model """
         # Prepare node and edge embeddings
        V, E, E_idx = self.model.features(X, L, mask)
        h_V = self.model.W_v(V)
        h_E = self.model.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.model.encoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = layer(h_V, h_EV, mask_V=mask, mask_attend=mask_attend)
        
        # Decoder alternates masked self-attention
        mask_attend = self.model._autoregressive_mask(E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)
        N_batch, N_nodes = X.size(0), X.size(1)
        h_S = torch.zeros_like(h_V)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device = self.device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V) for _ in range(len(self.model.decoder_layers))]
        for t in range(N_nodes):
            # Hidden layers
            E_idx_t = E_idx[:,t:t+1,:]
            h_E_t = h_E[:,t:t+1,:,:]
            
            # # remove cache
            # h_ES_enc_t = cat_neighbors_nodes(torch.zeros_like(h_S), h_E_t, E_idx_t)
            # enc_embedding = h_V
            # h_ESV_encoder_t = cat_neighbors_nodes(h_V, h_ES_enc_t, E_idx_t)
            # for l, layer in enumerate(self.model.decoder_layers):
            #     if l==0:
            #         h_ESV_t = cat_neighbors_nodes(enc_embedding, h_ES_t, E_idx_t)
            #         h_ESV_t = mask_bw[:,t:t+1,:,:] * h_ESV_t + mask_fw[:,t:t+1,:,:]*h_ESV_encoder_t 
            #         node_feaures = h_V[:,t:t+1,:]
            #     else:
            #         h_ESV_t = cat_neighbors_nodes(enc_embedding, h_ES_t, E_idx_t)
            #         h_ESV_t = mask_bw[:,t:t+1,:,:] * h_ESV_t + mask_fw[:,t:t+1,:,:]*h_ESV_encoder_t 
            #     node_feaures = layer(node_feaures, h_ESV_t, mask_V=mask[:,t:t+1])
            #     enc_embedding[:,t:t+1,::] = node_feaures

            # node_feaures = node_feaures.squeeze(1)
            # logits = self.model.W_out(node_feaures) / temperature
            # probs = F.softmax(logits, dim=-1)
            # S_t = torch.multinomial(probs, 1).squeeze(-1)
            
            # use cache
            h_ES_enc_t = cat_neighbors_nodes(torch.zeros_like(h_S), h_E_t, E_idx_t)
            h_ESV_encoder_t = mask_fw[:,t:t+1,:,:] * cat_neighbors_nodes(h_V, h_ES_enc_t, E_idx_t)
            
            for l, layer in enumerate(self.model.decoder_layers):
                # Updated relational features for future states
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t) # [batch, 1, K, 384]
                h_ESV_t = mask_bw[:,t:t+1,:,:] * h_ESV_decoder_t + h_ESV_encoder_t # [batch, 1, K, 384]
                
                h_V_t = h_V_stack[l][:,t:t+1,:] # [batch, 1 128]
                h_V_stack[l+1][:,t,:] = layer(
                    h_V_t, h_ESV_t, mask_V=mask[:,t:t+1]
                ).squeeze(1) # [1, 128]

            # Sampling step
            h_V_t = h_V_stack[-1][:,t,:]
            logits = self.model.W_out(h_V_t) / temperature
            probs = F.softmax(logits, dim=-1)
            S_t = torch.multinomial(probs, 1).squeeze(-1)
            
            # Update
            h_S[:,t,:] = self.model.W_s(S_t)
            S[:,t] = S_t
        return S