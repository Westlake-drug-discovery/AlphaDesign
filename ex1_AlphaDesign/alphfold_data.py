import os
import numpy as np
import torch
from torch.utils import data as torch_data
import _pickle as cPickle
import json
import random

class cached_property(object):
    """
    Descriptor (non-data) for building an attribute on-demand on first use.
    """
    def __init__(self, factory):
        """
        <factory> is called such: factory(instance) to build the attribute.
        """
        self._attr_name = factory.__name__
        self._factory = factory

    def __get__(self, instance, owner):
        # Build the attribute.
        attr = self._factory(instance)

        # Cache the value; hide ourselves.
        setattr(instance, self._attr_name, attr)

        return attr

class AlphaFold(torch_data.Dataset):
    def __init__(self, preprocess_path = './', mode = 'train', max_length = 500, limit_length = 1, joint_data=0):
        
        self.preprocess_path = preprocess_path
        self.max_length = max_length
        self.limit_length = limit_length
        self.joint_data=joint_data
        
        if mode == 'train':
            self.data = self.cache_data['train']
        
        if mode == 'valid':
            self.data = self.cache_data['valid']
        
        if mode == 'test':
            self.data = self.cache_data['test']
        
        if mode == 'all':
            self.data = self.cache_data['train'] + self.cache_data['valid'] + self.cache_data['test']
        
        self.lengths = np.array([ len(sample['seq']) for sample in self.data])
        self.max_len = np.max(self.lengths)
        self.min_len = np.min(self.lengths)

    def get_data(self, preprocess_path):
        if not os.path.exists(preprocess_path):
            raise "no such file:{} !!!".format(preprocess_path)
        else:
            name = preprocess_path.split('/')[-1]
            data_ = cPickle.load(open(preprocess_path+'/data_{}.pkl'.format(name),'rb'))
            score_ = cPickle.load(open(preprocess_path+'/data_{}_score.pkl'.format(name),'rb'))
            for i in range(len(data_)):
                data_[i]['score'] = score_[i]['res_score']
        
        data = []
        for temp in data_:
            if self.limit_length:
                if 30<len(temp['seq']) and len(temp['seq']) < self.max_length:
                    # 'title', 'seq', 'CA', 'C', 'O', 'N'
                    data.append(temp)
            else:
                data.append(temp)
        
        if self.limit_length:
            split_name = 'split'
        else:
            split_name = 'splitF'

        # split_name = 'split'
        if not os.path.exists(preprocess_path+'/{}.json'.format(split_name)):
            split = list(range(len(data)))
            random.shuffle(split)
            
            N = len(data)
            train_idx = split[:int(0.9*N)]
            valid_idx = split[int(0.9*N):int(0.9*N)+100]
            test_idx = split[int(0.9*N)+100:]
            
            split = {'train':train_idx, 'valid':valid_idx, 'test':test_idx}
            json.dump(split, open(preprocess_path+'/{}.json'.format(split_name),'w'))
        else:
            split = json.load(open(preprocess_path+'/{}.json'.format(split_name),'r'))
        
        data_dict = {'train':[ data[i] for i in split['train'] ],
                     'valid':[ data[i] for i in split['valid'] ],
                     'test':[ data[i] for i in split['test'] ]}
        return data_dict
    
    @cached_property
    def cache_data(self):
        preprocess_path = self.preprocess_path
        if self.joint_data:
            root_path = '/'+os.path.join(*preprocess_path.split('/')[:-1])
            datanames = [dataname for dataname in os.listdir(root_path) if ('_v2' in dataname)]
            data_dict = {'train':[], 'valid':[], 'test':[]}
            for dataname in datanames:
                temp = self.get_data(os.path.join(root_path, dataname))
                data_dict['train'] += temp['train']
                data_dict['valid'] += temp['valid']
                data_dict['test'] += temp['test']
        else:
            data_dict = self.get_data(preprocess_path)
        return data_dict
    
    def change_mode(self, mode):
        self.data = self.cache_data[mode]
    
    def __len__(self):
        return len(self.data)
    
    def get_item(self, index):
        return self.data[index]

    def __getitem__(self, index):
        return self.data[index]


###################################################################################
################################### NIPS19 ########################################

class DataLoader_NIPS19(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0,
                 collate_fn=None, **kwargs):
        super(DataLoader_NIPS19, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn,**kwargs)

def featurize_NIPS19(batch, shuffle_fraction=0.):
    """ Pack and pad batch into torch tensors """
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    score = np.zeros([B, L_max])

    def shuffle_subset(n, p):
        n_shuffle = np.random.binomial(n, p)
        ix = np.arange(n)
        ix_subset = np.random.choice(ix, size=n_shuffle, replace=False)
        ix_subset_shuffled = np.copy(ix_subset)
        np.random.shuffle(ix_subset_shuffled)
        ix[ix_subset] = ix_subset_shuffled
        return ix

    # Build the batch
    for i, b in enumerate(batch):
        x = np.stack([b[c] for c in ['N', 'CA', 'C', 'O']], 1) # [#atom, 4, 3]
        
        l = len(b['seq'])
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, )) # [#atom, 4, 3]
        X[i,:,:,:] = x_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
        if shuffle_fraction > 0.:
            idx_shuffle = shuffle_subset(l, shuffle_fraction)
            S[i, :l] = indices[idx_shuffle]
            score[i,:l] = b['score'][idx_shuffle]
        else:
            S[i, :l] = indices
            score[i,:l] = b['score']

    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32) # atom mask
    numbers = np.sum(mask, axis=1).astype(np.int)
    S_new = np.zeros_like(S)
    score_new = np.zeros_like(score)
    X_new = np.zeros_like(X)+np.nan
    for i, n in enumerate(numbers):
        X_new[i,:n,::] = X[i][mask[i]==1]
        S_new[i,:n] = S[i][mask[i]==1]
        score_new[i,:n] = score[i][mask[i]==1]

    X = X_new
    S = S_new
    score = score_new
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    # Conversion
    S = torch.from_numpy(S).to(dtype=torch.long)
    score = torch.from_numpy(score).float()
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    return X, S, score, mask, lengths


###################################################################################
##################################### GVP #########################################
import torch_geometric
import torch_cluster
import torch.nn.functional as F
import math

def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


from collections.abc import Mapping, Sequence
from torch_geometric.data import Data, Batch
from torch.utils.data.dataloader import default_collate

class featurize_GVP:
    def __init__(self, num_positional_embeddings=16, top_k=30, num_rbf=16):
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}
    
    def featurize(self, batch):
        data_all = []
        for b in batch:
            coords = torch.tensor(np.stack([b[c] for c in ['N', 'CA', 'C', 'O']], 1))
            seq = torch.tensor([self.letter_to_num[a] for a in b['seq']])
        
            mask = torch.isfinite(coords.sum(dim=(1,2)))
            coords[~mask] = np.inf
            
            X_ca = coords[:, 1].float()
            edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)
            
            pos_embeddings = self._positional_embeddings(edge_index) # [E, 16]
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]] # [E, 3]
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf) # [E, 16]
            
            dihedrals = self._dihedrals(coords)  # [n,6]
            orientations = self._orientations(X_ca) # [n,2,3]
            sidechains = self._sidechains(coords) # [n,3]
            
            node_s = dihedrals.float() # [n,6]

            # node_v:local coordinate system
            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2).float() # [n, 3, 3]

            # pos_embeddings = torch.zeros_like(pos_embeddings) 
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1).float() # [E, 32]
            edge_v = _normalize(E_vectors).unsqueeze(-2).float() # [E, 1, 3]
            
            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,(node_s, node_v, edge_s, edge_v))
            
            data = torch_geometric.data.Data(x=X_ca, seq=seq,
                                            node_s=node_s, node_v=node_v,
                                            edge_s=edge_s, edge_v=edge_v,
                                            edge_index=edge_index, mask=mask)
            data_all.append(data)
        return data_all
    
    def _positional_embeddings(self, edge_index, 
                               num_embeddings=None,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]
     
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _dihedrals(self, X, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2]) 
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features
    
    def _orientations(self, X):
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec 
    

    def collate(self, batch):
        batch = self.featurize(batch)
        
        elem = batch[0]
        if isinstance(elem, Data):
            return Batch.from_data_list(batch)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))


import torch.utils.data as data
class BatchSampler(data.Sampler):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design.
    
    A `torch.utils.data.Sampler` which samples batches according to a
    maximum number of graph nodes.
    
    :param node_counts: array of node counts in the dataset to sample from
    :param max_nodes: the maximum number of nodes in any batch,
                      including batches of a single element
    :param shuffle: if `True`, batches in shuffled order
    '''
    def __init__(self, node_counts, max_nodes=3000, shuffle=True):
        self.node_counts = node_counts
        self.idx = [i for i in range(len(node_counts))  
                        if node_counts[i] <= max_nodes]
        self.shuffle = shuffle
        self.max_nodes = max_nodes
        self._form_batches()
    
    def _form_batches(self):
        self.batches = []
        if self.shuffle: random.shuffle(self.idx)
        idx = self.idx
        while idx:
            batch = []
            n_nodes = 0
            while idx and n_nodes + self.node_counts[idx[0]] <= self.max_nodes:
                next_idx, idx = idx[0], idx[1:]
                n_nodes += self.node_counts[next_idx]
                batch.append(next_idx)
            self.batches.append(batch)
    
    def __len__(self): 
        if not self.batches: self._form_batches()
        return len(self.batches)
    
    def __iter__(self):
        if not self.batches: 
            self._form_batches()
        for batch in self.batches: 
            yield batch



class DataLoader_GVP(torch.utils.data.DataLoader):
    def __init__(self, dataset, num_workers=0,
                 featurizer=None, max_nodes=3000, **kwargs):
        super(DataLoader_GVP, self).__init__(dataset, 
                                            batch_sampler = BatchSampler(node_counts = [ len(data['seq']) for data in dataset], max_nodes=max_nodes), 
                                            num_workers = num_workers, 
                                            collate_fn = featurizer.collate,
                                            **kwargs)
