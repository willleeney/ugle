# https://github.com/GRAND-Lab/SUBLIME
import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
import numpy as np
import copy
from fast_pytorch_kmeans import KMeans
from ugle.trainer import ugleTrainer
EOS = 1e-10
import types

class GCNConv_dense(nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConv_dense, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def init_para(self):
        self.linear.reset_parameters()

    def forward(self, input, A, sparse=False):
        hidden = self.linear(input)
        output = torch.matmul(A, hidden)
        return output


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, Adj, sparse):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(GCNConv_dense(in_channels, hidden_channels))
        for i in range(num_layers - 2):
            self.layers.append(GCNConv_dense(hidden_channels, hidden_channels))
        self.layers.append(GCNConv_dense(hidden_channels, out_channels))

        self.dropout = dropout
        self.dropout_adj_p = dropout_adj
        self.Adj = Adj
        self.Adj.requires_grad = False

        self.dropout_adj = nn.Dropout(p=dropout_adj)

    def forward(self, x):

        Adj = self.dropout_adj(self.Adj)

        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, Adj)
        return x


class GraphEncoder(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj):

        super(GraphEncoder, self).__init__()
        self.dropout = dropout
        self.dropout_adj_p = dropout_adj

        self.gnn_encoder_layers = nn.ModuleList()

        self.gnn_encoder_layers.append(GCNConv_dense(in_dim, hidden_dim))
        for _ in range(nlayers - 2):
            self.gnn_encoder_layers.append(GCNConv_dense(hidden_dim, hidden_dim))
        self.gnn_encoder_layers.append(GCNConv_dense(hidden_dim, emb_dim))

        self.dropout_adj = nn.Dropout(p=dropout_adj)

        self.proj_head = Sequential(Linear(emb_dim, proj_dim), ReLU(inplace=True),
                                    Linear(proj_dim, proj_dim))

    def forward(self, x, Adj_, branch=None):

        Adj = self.dropout_adj(Adj_)

        for conv in self.gnn_encoder_layers[:-1]:
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gnn_encoder_layers[-1](x, Adj)
        z = self.proj_head(x)
        return z, x


class GCL(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse=False):
        super(GCL, self).__init__()

        self.encoder = GraphEncoder(nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj)

    def forward(self, x, Adj_, branch=None):
        z, embedding = self.encoder(x, Adj_, branch)
        return z, embedding

    @staticmethod
    def calc_loss(x, x_aug, temperature=0.2, sym=True):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        if sym:
            loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

            loss_0 = - torch.log(loss_0).mean()
            loss_1 = - torch.log(loss_1).mean()
            loss = (loss_0 + loss_1) / 2.0
            return loss
        else:
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss_1 = - torch.log(loss_1).mean()
            return loss_1


class MLP_learner(nn.Module):
    def __init__(self, nlayers, isize, k, knn_metric, i, sparse, act):
        super(MLP_learner, self).__init__()

        self.layers = nn.ModuleList()
        if nlayers == 1:
            self.layers.append(nn.Linear(isize, isize))
        else:
            self.layers.append(nn.Linear(isize, isize))
            for _ in range(nlayers - 2):
                self.layers.append(nn.Linear(isize, isize))
            self.layers.append(nn.Linear(isize, isize))

        self.input_dim = isize
        self.output_dim = isize
        self.k = k
        self.knn_metric = knn_metric
        self.non_linearity = 'relu'
        self.param_init()
        self.i = i
        self.act = act

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                if self.act == "relu":
                    h = F.relu(h)
                elif self.act == "tanh":
                    h = F.tanh(h)
        return h

    def param_init(self):
        for layer in self.layers:
            layer.weight = nn.Parameter(torch.eye(self.input_dim))

def cal_similarity_graph(node_embeddings):
    similarity_graph = torch.mm(node_embeddings, node_embeddings.t())
    return similarity_graph


def apply_non_linearity(tensor, non_linearity, i):
    if non_linearity == 'elu':
        return F.elu(tensor * i - i) + 1
    elif non_linearity == 'relu':
        return F.relu(tensor)
    elif non_linearity == 'none':
        return tensor
    else:
        raise NameError('We dont support the non-linearity yet')


def symmetrize(adj):  # only for non-sparse
    return (adj + adj.T) / 2


def normalize(adj, mode, sparse=False):

    if mode == "sym":
        inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
        return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
    elif mode == "row":
        inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
        return inv_degree[:, None] * adj
    else:
        exit("wrong norm mode")


def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


def get_feat_mask(features, mask_rate):
    feat_node = features.shape[1]
    mask = torch.zeros(features.shape)
    samples = np.random.choice(feat_node, size=int(feat_node * mask_rate), replace=False)
    mask[:, samples] = 1
    return mask, samples


class gclloss_obj:
    def __init__(self, maskfeat_rate_anchor, maskfeat_rate_learner, contrast_batch_size, device, n_nodes):
        super(gclloss_obj, self).__init__()
        self.maskfeat_rate_anchor = maskfeat_rate_anchor
        self.maskfeat_rate_learner = maskfeat_rate_learner
        self.contrast_batch_size = contrast_batch_size
        self.device = device
        self.n_nodes = n_nodes

    def top_k(self, raw_graph, K):
        values, indices = raw_graph.topk(k=int(K), dim=-1)
        assert torch.max(indices) < raw_graph.shape[1]
        mask = torch.zeros(raw_graph.shape).to(self.device)
        mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.

        mask.requires_grad = False
        sparse_graph = raw_graph * mask
        return sparse_graph

    def graph_learner_forward(self, graph_learner, features):
        embeddings = graph_learner.internal_forward(features)
        embeddings = F.normalize(embeddings, dim=1, p=2)
        similarities = cal_similarity_graph(embeddings)
        similarities = self.top_k(similarities, graph_learner.k + 1)
        similarities = apply_non_linearity(similarities, graph_learner.non_linearity, graph_learner.i)
        return similarities

    def loss_gcl(self, model, graph_learner, features, anchor_adj):

        # view 1: anchor graph
        if self.maskfeat_rate_anchor:
            mask_v1, _ = get_feat_mask(features, self.maskfeat_rate_anchor)
            mask_v1 = mask_v1.to(self.device)
            features_v1 = features * (1 - mask_v1)
        else:
            features_v1 = features.copy()

        features_v1 = features_v1.to(self.device)
        z1, _ = model(features_v1, anchor_adj, 'anchor')

        # view 2: learned graph
        if self.maskfeat_rate_learner:
            mask, _ = get_feat_mask(features, self.maskfeat_rate_learner)
            mask = mask.to(self.device)
            features_v2 = features * (1 - mask)
        else:
            features_v2 = features.copy()

        learned_adj = self.graph_learner_forward(graph_learner, features)


        learned_adj = symmetrize(learned_adj)
        learned_adj = normalize(learned_adj, 'sym')

        z2, _ = model(features_v2, learned_adj, 'learner')

        # compute loss
        if self.contrast_batch_size:
            node_idxs = list(range(self.n_nodes))
            # random.shuffle(node_idxs)
            batches = split_batch(node_idxs, self.contrast_batch_size)
            loss = 0
            for batch in batches:
                weight = len(batch) / self.n_nodes
                loss += model.calc_loss(z1[batch], z2[batch]) * weight
        else:
            loss = model.calc_loss(z1, z2)

        return loss, learned_adj


    

class sublime_trainer(ugleTrainer):
    def preprocess_data(self, features, adjacency):

        anchor_adj_raw = torch.from_numpy(adjacency)
        anchor_adj = normalize(anchor_adj_raw, 'sym')
        features = torch.FloatTensor(features)
        anchor_adj = anchor_adj.float()

        return features, anchor_adj

    def training_preprocessing(self, args, processed_data):
        if args.learner_type == 'mlp':
            self.graph_learner = MLP_learner(args.nlayers, args.n_features, args.k, args.sim_function, args.i, args.sparse,
                                        args.activation_learner).to(self.device)

        self.model = GCL(nlayers=args.nlayers, in_dim=args.n_features, hidden_dim=args.hidden_dim,
                    emb_dim=args.rep_dim, proj_dim=args.proj_dim,
                    dropout=args.dropout, dropout_adj=args.dropedge_rate).to(self.device)

        optimizer_cl = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        optimizer_learner = torch.optim.Adam(self.graph_learner.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.optimizers = [optimizer_cl, optimizer_learner]
        self.gcl_obj = gclloss_obj(self.cfg.args.maskfeat_rate_anchor, 
                                   self.cfg.args.maskfeat_rate_learner, 
                                   self.cfg.args.contrast_batch_size,
                                   self.device,
                                   self.cfg.args.n_nodes)

        return

    def training_epoch_iter(self, args, processed_data):
        features, anchor_adj = processed_data

        loss, Adj = self.gcl_obj.loss_gcl(self.model, self.graph_learner, features, anchor_adj)

        # Structure Bootstrapping
        if (1 - args.tau) and (args.c == 0 or self.current_epoch % args.c == 0):
            anchor_adj = anchor_adj * args.tau + Adj.detach() * (1 - args.tau)

        processed_data = (features, anchor_adj)

        return loss, processed_data

    def test(self, processed_data):
        features, anchor_adj = processed_data

        self.model.eval()
        self.graph_learner.eval()
        with torch.no_grad():
            _, Adj = self.gcl_obj.loss_gcl(self.model, self.graph_learner, features, anchor_adj)
            _, embedding = self.model(features, Adj)
            embedding = embedding.squeeze(0)

            kmeans = kmeans = KMeans(n_clusters=self.cfg.args.n_clusters)
            preds = kmeans.fit_predict(embedding).cpu().numpy()

        return preds
