# inspired by https://github.com/PetarV-/DGI
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from fast_pytorch_kmeans import KMeans
import ugle
from ugle.trainer import ugleTrainer
from ugle.gnn_architecture import GCN, AvgReadout, Discriminator

class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse=True)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(seq2, adj, sparse=True)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq, adj, msk):
        h_1 = self.gcn(seq, adj, sparse=True)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()


class dgi_trainer(ugleTrainer):

    def preprocess_data(self, features, adjacency):
        adjacency = adjacency + sp.eye(adjacency.shape[0])
        adjacency = ugle.process.normalize_adj(adjacency)
        adj = ugle.process.sparse_mx_to_torch_sparse_tensor(adjacency)
        features = ugle.process.preprocess_features(features)
        features = torch.FloatTensor(features[np.newaxis])

        return adj, features

    def training_preprocessing(self, args, processed_data):
        self.model = DGI(args.n_features, args.hid_units, args.nonlinearity).to(self.device)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.loss_function = nn.BCEWithLogitsLoss()
        self.optimizers = [optimiser]

        return

    def training_epoch_iter(self, args, processed_data):
        adj, features = processed_data
        idx = np.random.permutation(args.n_nodes)
        shuf_fts = features[:, idx, :]
        lbl_1 = torch.ones(args.batch_size, args.n_nodes)
        lbl_2 = torch.zeros(args.batch_size, args.n_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)
        shuf_fts = shuf_fts.to(self.device)
        lbl = lbl.to(self.device)
        logits = self.model(features, shuf_fts, adj, None, None, None)
        loss = self.loss_function(logits, lbl)

        return loss, None

    def test(self, processed_data):
        adj, features = processed_data
        with torch.no_grad():
            embeds, _ = self.model.embed(features, adj, None)
            embeds = embeds.squeeze(0)

        kmeans = KMeans(n_clusters=self.cfg.args.n_clusters)
        preds = kmeans.fit_predict(embeds).cpu().numpy()

        return preds
