from cmath import log
import logging
import os
import json
import nni
import numpy as np
from tqdm import tqdm
from shutil import copyfile
import shutil
import torch
import re
import copy  
from src.utils.tools import cuda
import warnings
warnings.filterwarnings("ignore")



def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av

class Exp:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:{}'.format(0))
        self.batch_size = args.batch_size

        self.path = self.args.res_dir+'/{}'.format(self.args.ex_name)
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        files = [f for f in os.listdir('/'+os.path.join(*self.args.res_dir.split('/')[:-1])) if re.search('\.py', f)]
        for file in files:
            copyfile('/' + os.path.join(*self.path.split('/')[:-2]) + '/{}'.format(file), self.path +'/{}'.format(file))
        
        
        root = '/' + os.path.join(*self.path.split('/')[:-3])
        if self.args.method == 'NIPS19':
            shutil.rmtree(self.path+'/NIPS19',ignore_errors=True)
            shutil.copytree(root+'/src/NIPS19', self.path+'/NIPS19')
        
        if self.args.method == 'SGNN':
            shutil.rmtree(self.path+'/NIPS19',ignore_errors=True)
            shutil.copytree(root+'/src/NIPS19', self.path+'/NIPS19')
            
        if self.args.method == 'GVP':
            shutil.rmtree(self.path+'/GVP',ignore_errors=True)
            shutil.copytree(root+'/src/GVP', self.path+'/GVP')
        
        if self.args.method == 'AlphaDesign':
            shutil.rmtree(self.path+'/AlphaDesign',ignore_errors=True)
            shutil.copytree(root+'/src/AlphaDesign', self.path+'/AlphaDesign')

        self.checkpoints_path = os.path.join(self.path, 'checkpoints')
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)

        sv_param = os.path.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
                            filename=self.path+'/log.log',#'log/{}_{}_{}.log'.format(args.gcn_type,args.graph_type,args.order_list)
                            filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                            #a是追加模式，默认如果不写的话，就是追加模式
                            format='%(asctime)s - %(message)s'#日志格式
                            )
        logging.info(args)
        self.get_data()
        self.model = self._build_model().to(self.device)
        
        if self.args.epoch_s>0:
            self._load(self.args.epoch_s-1)

    def _build_model(self):
        if self.args.method == 'NIPS19':
            from src.NIPS19.model import NIPS19_model
            model = NIPS19_model(self.args, use_mpnn=False)
        
        if self.args.method == 'SGNN':
            from src.NIPS19.model import NIPS19_model
            model = NIPS19_model(self.args, use_mpnn=True)
        
        if self.args.method == 'GVP':
            from src.GVP.model import CPDModel
            node_dim = (100, 16)
            edge_dim = (32, 1)
            model = CPDModel((6, 3), node_dim, (32, 1), edge_dim)
        
        if self.args.method == 'AlphaDesign':
            from src.AlphaDesign.ADesign6 import ADesign
            model = ADesign(
                node_features=self.args.hidden,
                edge_features=self.args.hidden, 
                hidden_dim=self.args.hidden,
                dropout=self.args.dropout,
                k_neighbors=self.args.top_k,
                num_encoder_layers=5
            )
        
        return model
    
    def get_data(self):
        args = self.args
        from alphfold_data import AlphaFold
        preprocess_path = os.path.join(args.preprocess_path, args.data_name)
        self.train_set = AlphaFold(preprocess_path, mode = 'train', limit_length=args.limit_length, joint_data=args.joint_data) 
        self.valid_set = copy.copy(self.train_set)
        self.valid_set.change_mode('valid')
        self.test_set = copy.copy(self.train_set)
        self.test_set.change_mode('test')

        print('train:{}\tvalid:{}\ttest:{}'.format(len(self.train_set), len(self.valid_set), len(self.test_set)))
        logging.info('train:{}\tvalid:{}\ttest:{}'.format(len(self.train_set), len(self.valid_set), len(self.test_set)))
        
        if args.method == 'NIPS19' or args.method == 'SGNN':
            from alphfold_data import featurize_NIPS19 as collate_fn
            from alphfold_data import DataLoader_NIPS19
            self.featurizer = collate_fn
            self.train_loader = DataLoader_NIPS19(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
            self.valid_loader = DataLoader_NIPS19(self.valid_set, batch_size=self.batch_size, shuffle=False,  num_workers=4, collate_fn=collate_fn)
            self.test_loader = DataLoader_NIPS19(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
            
        if args.method == 'GVP':
            from alphfold_data import featurize_GVP, DataLoader_GVP
            self.featurizer = featurize_GVP()
            self.train_loader = DataLoader_GVP(self.train_set, num_workers=8, featurizer=self.featurizer, max_nodes=args.max_nodes)
            self.valid_loader = DataLoader_GVP(self.valid_set, num_workers=8, featurizer=self.featurizer, max_nodes=args.max_nodes)
            self.test_loader = DataLoader_GVP(self.test_set, num_workers=8, featurizer=self.featurizer, max_nodes=args.max_nodes)
        
        if args.method == 'AlphaDesign':
            from alphfold_data import featurize_NIPS19 as collate_fn
            from alphfold_data import DataLoader_NIPS19
            self.featurizer = collate_fn
            self.train_loader = DataLoader_NIPS19(self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
            self.valid_loader = DataLoader_NIPS19(self.valid_set, batch_size=self.batch_size, shuffle=False,  num_workers=4, collate_fn=collate_fn)
            self.test_loader = DataLoader_NIPS19(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    def _select_optimizer(self):
        args = self.args
        self.model_optim = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        steps_per_epoch = len(self.train_loader.dataset)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.model_optim, max_lr=self.args.lr, steps_per_epoch=steps_per_epoch, epochs=self.args.epoch_e)
        return self.model_optim

    def _save(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.checkpoints_path, str(epoch) + '.pth'))

    def _load(self,epoch):
        self.model.load_state_dict(torch.load(os.path.join(self.checkpoints_path, str(epoch) + '.pth')))
    
    def train(self, args):
        from src.utils.tools import EarlyStopping
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()

        for epoch in range(args.epoch_s, args.epoch_e):
            train_loss = []
            self.model.train()
            train_pbar = tqdm(self.train_loader)
            max_length = 0
            for batch in train_pbar:
                model_optim.zero_grad()
                
                if args.method == 'NIPS19' or args.method == 'SGNN':
                    X, S, mask, lengths = batch
                    X, S, mask, lengths = cuda((X, S, mask, lengths), device = self.device)
                    log_probs = self.model(X, S, lengths, mask)
                    _, loss = loss_smoothed(S, log_probs, mask, weight=self.args.smoothing)
                    loss.backward()
                
                if args.method == 'GVP':
                    batch = batch.to(self.device)
                    h_V = (batch.node_s, batch.node_v)
                    h_E = (batch.edge_s, batch.edge_v)
                    logits = self.model(h_V, batch.edge_index, h_E, seq=batch.seq)
                    logits, seq = logits[batch.mask], batch.seq[batch.mask]
                    loss_fn = torch.nn.CrossEntropyLoss()
                    loss = loss_fn(logits, seq)
                    loss.backward()
                
                if args.method == 'AlphaDesign':
                    X, S, score, mask, lengths = cuda(batch, device = self.device)
                    if max_length<lengths.max():
                        max_length=lengths.max()
                    X, S, score, h_V, h_E, E_idx, batch_id = self.model._get_features(S, score, X=X, mask=mask)
                    
                    log_probs, log_probs0 = self.model( h_V, h_E, E_idx, batch_id)

                    loss_fn = torch.nn.CrossEntropyLoss()
                    loss = loss_fn(log_probs, S)
                    loss0 = loss_fn(log_probs0, S)

                    (loss+loss0).backward()

                train_loss.append(loss.item())
                train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))
                
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                model_optim.step()
                self.scheduler.step()
                
            
            if epoch % args.log_step == 0:
                self._save(epoch)
                with torch.no_grad():
                    perplexity = self.evaluate(self.test_loader,'valid')

                print("Epoch: {0} | perplexity: {1:.7f}\n".format(epoch, perplexity))
                logging.info("Epoch: {0} | perplexity: {1:.7f}\n".format(epoch, perplexity))
                early_stopping(perplexity, self.model, self.path)

            if early_stopping.early_stop:
                print("Early stopping")
                logging.info("Early stopping")
                break
            
            torch.cuda.empty_cache()
            
        best_model_path = self.path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model
    
    @torch.no_grad()
    def evaluate(self, dataloader, name):
        self.model.eval()
        validation_sum, validation_weights = 0., 0.
        
        for count, batch in enumerate(dataloader):
            with torch.no_grad():
                if self.args.method == 'NIPS19'  or self.args.method == 'SGNN':
                    X, S, mask, lengths = cuda(batch, device = self.device)
                    log_probs = self.model(X, S, lengths, mask)
                    loss, loss_av = loss_nll(S, log_probs, mask)

                if self.args.method == 'GVP':
                    batch = batch.to(self.device)
                    h_V = (batch.node_s, batch.node_v)
                    h_E = (batch.edge_s, batch.edge_v)
                    logits = self.model(h_V, batch.edge_index, h_E, seq=batch.seq)
                    logits, seq = logits[batch.mask], batch.seq[batch.mask]
                    loss_fn = torch.nn.CrossEntropyLoss()
                    loss = loss_fn(logits, seq)
                    mask = batch.mask
                    
                if self.args.method == 'AlphaDesign':
                    X, S, mask, lengths = cuda(batch, device = self.device)
                    X, S, h_V, h_E, E_idx, batch_id = self.model._get_features(S, X=X, mask=mask)
                    log_probs, log_probs0 = self.model(h_V, h_E, E_idx, batch_id)
                    loss, loss_av = loss_nll_flatten(S, log_probs)
                    mask = torch.ones_like(loss)
                    
                # Accumulate
                validation_sum += torch.sum(loss * mask).cpu().data.numpy()
                validation_weights += torch.sum(mask).cpu().data.numpy()
        
        validation_loss = validation_sum / validation_weights
        validation_perplexity = np.exp(validation_loss)

        if name == 'valid':
            nni.report_intermediate_result(validation_perplexity)
            pass
        if name == 'test':
            recovery = self.sample_all(self.test_set)
            print('recovery:{}'.format(recovery))
            logging.info("test recovery: {0:.7f}\n".format(recovery))
            nni.report_final_result(recovery)
        return validation_perplexity

    @torch.no_grad()
    def sample_all(self, dataset, temperature=0.1, n_samples=1):
        recovery = []
        count = 0
        for protein in tqdm(dataset):
            if self.args.method == 'NIPS19'  or self.args.method == 'SGNN':
                from src.utils import cuda
                protein = self.featurizer([protein])
                X, S, mask, lengths = cuda(protein, device = self.device)
                sample = self.model.sample(X=X, L=lengths, mask=mask, temperature=0.1)
                recovery_ = sample.eq(S).float().mean().cpu().numpy()
            
            if self.args.method == 'GVP':
                protein = self.featurizer.collate([protein])
                protein = protein.to(self.device)
                recovery_ = self.model.test_recovery(protein)
            
            if self.args.method == 'AlphaDesign':
                from src.utils import cuda
                from alphfold_data import featurize_NIPS19 as collate_fn
                protein = collate_fn([protein])
                X, S, mask, lengths = cuda(protein, device = self.device)
                X, S, h_V, h_E, E_idx, batch_id = self.model._get_features(S, X=X, mask=mask)
                log_probs, log_probs0 = self.model(h_V, h_E, E_idx, batch_id)
                S_pred = torch.argmax(log_probs, dim=1)
                recovery_ = (S_pred == S).float().mean().cpu().numpy()
            
            recovery.append(recovery_)
            count+=1

        
        recovery = np.median(recovery)
        return recovery

def loss_smoothed(S, log_probs, mask, weight=0.1):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S, num_classes=20).float() # [4, 463] --> [4, 463, 20]

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True) # [4, 463, 20]/[4, 463, 1] --> [4, 463, 20]

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av

def loss_smoothed_faltten(S, log_probs, weight=0.1):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S, num_classes=20).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = loss.mean()
    return loss, loss_av

def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av

def loss_nll_flatten(S, log_probs):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(log_probs, S)
    loss_av = loss.mean()
    return loss, loss_av