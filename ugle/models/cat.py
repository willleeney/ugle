import ugle
import scipy.sparse as sp
from ugle.trainer import ugleTrainer
import torch.nn as nn
import torch
from ugle.gnn_architecture import GCN, AvgReadout, Discriminator
import math
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch.nn.functional as F


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return torch.mean(2 - 2 * (x * y).sum(dim=-1))


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


        self.student_gcn = GCN(args.architecture, args.n_clusters, act=act)
        self.teacher_gcn = deepcopy(self.student_gcn)
        set_requires_grad(self.teacher_gcn, False)
        self.teacher_ema_updater = EMA(self.args.beta, self.args.max_epoch)


        #self.con_loss_fn = nn.CrossEntropyLoss()

        #self.aug_gcn = deepcopy(self.gcn)
        #set_requires_grad(self.aug_gcn, False)
        #self.teacher_ema_updater = EMA(self.args.beta, self.args.max_epoch)

        # # sigmoid decoder
        # self.sigm = nn.Sigmoid()
        # self.recon_loss = nn.BCELoss()
        # self.recon_loss_reg = args.recon_loss_reg

        # # contrastive architecture
        # self.read = AvgReadout()
        # self.disc = Discriminator(args.architecture)
        # self.contrastive_loss = nn.BCEWithLogitsLoss()
        self.con_loss_reg = args.con_loss_reg

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data, gain=math.sqrt(2))
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

        self.transform.apply(init_weights)

        return
    
    # def bce_pytorch_geometric(pred_graph, real_graph):
    #     loss = torch.FloatTensor(0.)
    #     # go thru all pred_graph 
    #     # if there exists a link there then do: true * log (pred) + (1-true)*log(1-pred)
    #     # else just do log(1-pred)
    #     # sum up all of these and then div by the constant below
    #     # might need fancy index strat
    #     return - loss / pred_graph.shape(0)
    
    def update_moving_average(self):
        assert self.teacher_gcn is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_gcn, self.student_gcn)


    def forward(self, graph, graph_normalised, features, aug_features, lbl, dense_graph):

        self.update_moving_average()

        gcn_out = self.gcn(features, graph_normalised, sparse=True)
        assignments = self.student_gcn(gcn_out, graph_normalised, sparse=True).squeeze(0)
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

        # sigmoid decoder 
        #adj_rec = self.sigm(torch.matmul(gcn_out.squeeze(0), gcn_out.squeeze(0).t()))
        # reconstruction loss
        #loss = self.recon_loss_reg * self.recon_loss(adj_rec.view(-1), dense_graph.view(-1))
        #feature_rec = self.sigm(self.decoder_gcn(gcn_out, graph_normalised, sparse=True))
        #loss += self.recon_loss_reg * self.recon_loss(feature_rec.view(-1), features.view(-1))

        # contrastive architecture
        with torch.no_grad(): 
            assingments_hat = F.softmax(self.teacher_gcn(self.gcn(aug_features, graph_normalised, sparse=True), graph_normalised, sparse=True).squeeze(0))
            
        loss += self.con_loss_reg * loss_fn(assingments_hat, assignments)

        #c = self.sigm(self.read(gcn_out))
        #ret = self.disc(c, gcn_out, aug_out)
        # contrastive loss function
        #loss += self.con_loss_reg * self.contrastive_loss(ret, lbl)

        return loss

    def embed(self, graph_normalised, features):
        gcn_out = self.gcn(features, graph_normalised, sparse=True)
        assignments = self.student_gcn(gcn_out, graph_normalised, sparse=True).squeeze(0)
        assignments = F.softmax(assignments, dim=1)

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

        # add corrupted features 
        idx = torch.randperm(self.cfg.args.n_nodes)
        aug_features = features[idx, :]

        # add 'ground-truth' labels
        lbl_1 = torch.ones(1, self.cfg.args.n_nodes)
        lbl_2 = torch.zeros(1, self.cfg.args.n_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)
        aug_features = aug_features.to(self.device)
        lbl = lbl.to(self.device)

        dense_graph = graph_normalised.to_dense().to(self.device)

        return graph, graph_normalised, features, aug_features, lbl, dense_graph

    def training_preprocessing(self, args, processed_data):

        self.model = CAT(args).to(self.device)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.optimizers = [optimiser]

        return

    def training_epoch_iter(self, args, processed_data):
        graph, graph_normalised, features, aug_features, lbl, dense_graph = processed_data
        loss = self.model(graph, graph_normalised, features, aug_features, lbl, dense_graph)
        
        return loss, None

    def test(self, processed_data):
        # need to check if this is still right?
        _, graph_normalised, features, _, _, _ = processed_data
        with torch.no_grad():
            assignments = self.model.embed(graph_normalised, features)
            preds = assignments.detach().cpu().numpy().argmax(axis=1)

        return preds



