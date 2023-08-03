# https://github.com/kavehhassani/mvgrl
import torch
import torch.nn as nn
import ugle
import scipy.sparse as sp
import numpy as np
from fast_pytorch_kmeans import KMeans
from sklearn.preprocessing import MinMaxScaler
from ugle.trainer import ugleTrainer
from ugle.gnn_architecture import GCN, AvgReadout, mvgrl_Discriminator
from ugle.process import sparse_mx_to_torch_sparse_tensor


class Model(nn.Module):
    def __init__(self, n_in, n_h, act):
        super(Model, self).__init__()
        self.gcn1 = GCN(n_in, n_h, act)
        self.gcn2 = GCN(n_in, n_h, act)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = mvgrl_Discriminator(n_h)

    def forward(self, seq1, seq2, adj, diff, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn1(seq1, adj, sparse)
        c_1 = self.read(h_1, msk)
        c_1 = self.sigm(c_1)

        h_2 = self.gcn2(seq1, diff, sparse)
        c_2 = self.read(h_2, msk)
        c_2 = self.sigm(c_2)

        h_3 = self.gcn1(seq2, adj, sparse)
        h_4 = self.gcn2(seq2, diff, sparse)

        ret = self.disc(c_1, c_2, h_1, h_2, h_3, h_4, samp_bias1, samp_bias2)

        return ret, h_1, h_2

    def embed(self, seq, adj, diff, sparse, msk):
        h_1 = self.gcn1(seq, adj, sparse)
        c = self.read(h_1, msk)

        h_2 = self.gcn2(seq, diff, sparse)
        return (h_1 + h_2).detach(), c.detach()


class mvgrl_trainer(ugleTrainer):
    def preprocess_data(self, features, adjacency):
        epsilons = [1e-5, 1e-4, 1e-3, 1e-2]

        diff_adj = ugle.process.compute_ppr(adjacency)
        avg_degree = np.sum(adjacency) / adjacency.shape[0]
        epsilon = epsilons[np.argmin([abs(avg_degree - np.argwhere(diff_adj >= e).shape[0] / diff_adj.shape[0])
                                      for e in epsilons])]

        diff_adj[diff_adj < epsilon] = 0.0
        scaler = MinMaxScaler()
        scaler.fit(diff_adj)
        diff_adj = scaler.transform(diff_adj)

        features = ugle.process.preprocess_features(features)
        adjacency = ugle.process.normalize_adj(adjacency + sp.eye(adjacency.shape[0])).toarray()

        return features, adjacency, diff_adj

    def training_preprocessing(self, args, processed_data):
        features, adj, diff_adj = processed_data

        if adj.shape[-1] < args.sample_size:
            args.sample_size = int(np.floor(adj.shape[-1] / 100.0) * 100)

        self.model = Model(args.n_features, args.hid_units, args.activation).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.optimizers = [optimizer]
        self.loss_function = nn.BCEWithLogitsLoss()

        return

    def training_epoch_iter(self, args, processed_data):

        features, adj, diff_adj = processed_data

        if adj.shape[-1] < self.cfg.args.sample_size:
            self.cfg.args.sample_size = int(np.floor(adj.shape[-1] / 100.0) * 100)

        lbl_1 = torch.ones(self.cfg.args.batch_size, self.cfg.args.sample_size * 2)
        lbl_2 = torch.zeros(self.cfg.args.batch_size, self.cfg.args.sample_size * 2)
        lbl = torch.cat((lbl_1, lbl_2), 1).to(self.device)
        idx = np.random.randint(0, adj.shape[-1] - args.sample_size + 1, args.batch_size)

        ba, bd, bf = [], [], []
        for i in idx:
            ba.append(adj[i: i + args.sample_size, i: i + args.sample_size])
            bd.append(diff_adj[i: i + args.sample_size, i: i + args.sample_size])
            bf.append(features[i: i + args.sample_size])

        ba = np.asarray(ba).reshape(args.batch_size, args.sample_size, args.sample_size)
        bd = np.array(bd).reshape(args.batch_size, args.sample_size, args.sample_size)
        bf = np.array(bf).reshape(args.batch_size, args.sample_size, args.n_features)

        if args.sparse:
            ba = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ba))
            bd = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(bd))
        else:
            ba = torch.FloatTensor(ba)
            bd = torch.FloatTensor(bd)

        bf = torch.FloatTensor(bf)
        idx = np.random.permutation(args.sample_size)
        shuf_fts = bf[:, idx, :].to(self.device)
        ba = ba.to(self.device)
        bd = bd.to(self.device)
        bf = bf.to(self.device)

        logits, _, _ = self.model(bf, shuf_fts, ba, bd, args.sparse, None, None, None)

        loss = self.loss_function(logits, lbl)

        return loss, None

    def test(self, processed_data):
        features, adj, diff_adj = processed_data

        if self.cfg.args.sparse:
            adj = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj))
            diff_adj = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(diff_adj))

        features = torch.FloatTensor(features[np.newaxis]).to(self.device)
        adj = torch.FloatTensor(adj[np.newaxis]).to(self.device)
        diff_adj = torch.FloatTensor(diff_adj[np.newaxis]).to(self.device)

        embeds, _ = self.model.embed(features, adj, diff_adj, self.cfg.args.sparse, None)
        embeds = embeds.squeeze(0)

        kmeans = kmeans = KMeans(n_clusters=self.cfg.args.n_clusters)
        preds = kmeans.fit_predict(embeds).cpu().numpy()

        return preds
