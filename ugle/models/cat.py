import ugle
import scipy.sparse as sp
from ugle.trainer import ugleTrainer
import torch.nn as nn
import torch
from ugle.gnn_architecture import GCN
import math
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch.nn.functional as F
import wandb

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return torch.sum(2 - 2 * (x * y).sum(dim=-1))


class EMA:
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        beta = 1 - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        return old * beta + (1 - beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class CAT(nn.Module):
    def __init__(self,
                 args,
                 act='selu',
                 do_unpooling=False):
        """Initializes the layer with specified parameters."""
        super(CAT, self).__init__()
        self.args = args
        self.n_clusters = args.n_clusters
        self.cluster_size_regularization = args.cluster_size_regularization
        self.dropout_rate = args.dropout_rate
        self.do_unpooling = do_unpooling
        self.gcn = GCN(args.n_features, args.architecture, act=act)
        self.transform = nn.Sequential(OrderedDict([
            ('layer1', nn.Linear(args.architecture, args.n_clusters)),
            ('dropout', nn.Dropout(args.dropout_rate)),
        ]))

        self.predict_contrastive = nn.Sequential(OrderedDict([
            ('layer1', nn.Linear(args.architecture, args.architecture)),
            ('dropout', nn.Dropout(args.dropout_rate)),
        ]))

        self.teacher_gcn = deepcopy(self.gcn)
        set_requires_grad(self.teacher_gcn, False)
        self.teacher_ema_updater = EMA(self.args.beta, self.args.max_epoch)
        self.con_loss_reg = args.con_loss_reg

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data, gain=math.sqrt(2))
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

        self.transform.apply(init_weights)
        self.predict_contrastive.apply(init_weights)

        self.epoch_counter = 0 
        wandb.init(project='cat', entity='phd-keep-learning')

        return
    
    def update_moving_average(self):
        assert self.teacher_gcn is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_gcn, self.gcn)


    def forward(self, graph, graph_normalised, features, lbl, dense_graph):
        
        if self.epoch_counter % 100 == 0:
            self.idx = torch.randperm(self.args.n_nodes)
            self.idx2 = torch.randperm(self.args.n_nodes)

        # add corrupted features 
        aug_features = features[self.idx, :].to(features.device)
        aug_features2 = features[self.idx2, :].to(features.device)

        self.update_moving_average()

        gcn_out = self.gcn(features, graph_normalised, sparse=True)
        assignments = self.transform(gcn_out).squeeze(0)
        assignments = F.softmax(assignments, dim=1)

        n_edges = graph._nnz()
        degrees = torch.sparse.sum(graph, dim=0)._values().unsqueeze(1)
        graph_pooled = torch.spmm(torch.spmm(graph, assignments).T, assignments)
        normalizer_left = torch.spmm(assignments.T, degrees)
        normalizer_right = torch.spmm(assignments.T, degrees).T
        normalizer = torch.spmm(normalizer_left, normalizer_right) / 2 / n_edges
        spectral_loss = - torch.trace(graph_pooled - normalizer) / 2 / n_edges
        loss = spectral_loss
       
        cluster_sizes = torch.sum(assignments, dim=0)
        cluster_loss = torch.norm(cluster_sizes) / self.args.n_nodes * math.sqrt(float(self.n_clusters)) - 1
        cluster_loss *= self.cluster_size_regularization
        loss += cluster_loss

        # contrastive architecture
        pred_ass = self.predict_contrastive(self.gcn(aug_features, graph_normalised, sparse=True))
        with torch.no_grad(): 
            assingments_hat = self.teacher_gcn(aug_features2, graph_normalised, sparse=True)
        
        con_loss = self.con_loss_reg * loss_fn(assingments_hat.squeeze(0), pred_ass.squeeze(0))
        loss += con_loss

        wandb.log({'con_loss': con_loss, 
                   'spectral_loss': spectral_loss,
                   'cluster_loss': cluster_loss}, commit=True)
        
        self.epoch_counter += 1
        return loss

    def embed(self, graph_normalised, features):
        gcn_out = self.gcn(features, graph_normalised, sparse=True)
        assignments = self.transform(gcn_out).squeeze(0)
        assignments = nn.functional.softmax(assignments, dim=1)

        return assignments


class cat_trainer(ugleTrainer):

    def preprocess_data(self, features, adjacency):
        features = torch.FloatTensor(features)
        features[features != 0.] = 1.

        adjacency = adjacency + sp.eye(adjacency.shape[0])
        adj_label = adjacency.copy()
        adjacency = ugle.process.normalize_adj(adjacency)

        graph_normalised = ugle.process.sparse_mx_to_torch_sparse_tensor(adjacency)

        adj_label = sp.coo_matrix(adj_label)
        graph = ugle.process.sparse_mx_to_torch_sparse_tensor(adj_label)

        # add 'ground-truth' labels
        lbl_1 = torch.ones(1, self.cfg.args.n_nodes)
        lbl_2 = torch.zeros(1, self.cfg.args.n_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)

        dense_graph = graph_normalised.to_dense()

        return graph, graph_normalised, features, lbl, dense_graph

    def training_preprocessing(self, args, processed_data):

        self.model = CAT(args).to(self.device)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.optimizers = [optimiser]

        return

    def training_epoch_iter(self, args, processed_data):
        graph, graph_normalised, features, lbl, dense_graph = processed_data
        loss = self.model(graph, graph_normalised, features, lbl, dense_graph)
        
        return loss, None

    def test(self, processed_data):
        _, graph_normalised, features, _, _ = processed_data
        with torch.no_grad():
            assignments = self.model.embed(graph_normalised, features)
            preds = assignments.detach().cpu().numpy().argmax(axis=1)

        return preds
