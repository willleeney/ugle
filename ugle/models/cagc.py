# https://github.com/wangtong627/CAGC/
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from ugle.trainer import ugleTrainer
import numpy as np
from sklearn import cluster
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
import torch.nn.functional as F


def knn_fast(X, k):
    X = F.normalize(X, dim=1, p=2)
    similarities = torch.mm(X, X.t())
    vals, inds = similarities.topk(k=k + 1, dim=-1)
    return inds


def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def semi_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float):
    f = lambda x: torch.exp(x / tau)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(
        between_sim.diag()
        / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))


def instanceloss(z1: torch.Tensor, z2: torch.Tensor, tau: float, mean: bool = True):
    l1 = semi_loss(z1, z2, tau)
    l2 = semi_loss(z2, z1, tau)
    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()
    return ret


def knbrsloss(H, k, n_nodes, tau_knbrs, device):
    indices = knn_fast(H, k)
    f = lambda x: torch.exp(x / tau_knbrs)
    refl_sim = f(sim(H, H))
    ind2 = indices[:, 1:]
    V = torch.gather(refl_sim, 1, ind2)
    ret = -torch.log(
        V.sum(1) / (refl_sim.sum(1) - refl_sim.diag()))
    ret = ret.mean()
    return ret


def post_proC(C, K, d=6, alpha=8):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5 * (C + C.T)
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L


def thrC(C, ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while stop == False:
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model, k: int = 2, skip=False):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.skip = skip
        if not self.skip:
            self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
            for _ in range(1, k - 1):
                self.conv.append(base_model(1 * out_channels, 1 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if not self.skip:
            for i in range(self.k):
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]


class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model, k: int = 2, skip=False):
        super(Decoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.skip = skip
        if not self.skip:
            self.conv = [base_model(in_channels, 2 * in_channels).jittable()]
            for _ in range(1, k - 1):
                self.conv.append(base_model(1 * in_channels, 1 * in_channels))
            self.conv.append(base_model(2 * in_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, in_channels)]
            for _ in range(1, k - 1):
                self.conv = [base_model(in_channels, in_channels)]
            self.conv.append(base_model(in_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)
            self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if not self.skip:
            for i in range(self.k):
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]


class Model(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, num_sample: int, device):
        super(Model, self).__init__()
        self.device = device
        self.n = num_sample
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder
        self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(self.n, self.n, dtype=torch.float32),
                                        requires_grad=True).to(self.device)

    def forward(self, x, edge_index):
        # self expression layer, reshape to vectors, multiply Coefficient, then reshape back
        H = self.encoder(x, edge_index)
        CH = torch.matmul(self.Coefficient, H)
        X_ = self.decoder(CH, edge_index)

        return H, CH, self.Coefficient, X_


class cagc_trainer(ugleTrainer):
    def preprocess_data(self, features, adjacency):
        adj_label = sp.coo_matrix(adjacency)
        adj_label = adj_label.todok()

        outwards = [i[0] for i in adj_label.keys()]
        inwards = [i[1] for i in adj_label.keys()]

        adj = torch.from_numpy(np.array([outwards, inwards], dtype=int))

        data = torch.FloatTensor(features)

        self.cfg.args.alpha = max(0.4 - (self.cfg.args.n_clusters - 1) / 10 * 0.1, 0.1)
        self.cfg.hypersaved_args.alpha = self.cfg.args.alpha

        return data, adj

    def training_preprocessing(self, args, processed_data):
        activation = nn.PReLU()
        encoder = Encoder(args.n_features, args.num_hidden, activation,
                          base_model=GATConv, k=args.num_layers).to(self.device)

        decoder = Decoder(args.num_hidden, args.n_features, activation,
                          base_model=GATConv, k=args.num_layers).to(self.device)

        self.model = Model(encoder, decoder, args.n_nodes, self.device).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        self.optimizers = [optimizer]

        return

    def training_epoch_iter(self, args, processed_data):
        data, adj = processed_data
        H, CH, Coefficient, X_ = self.model(data, adj)
        loss_knbrs = knbrsloss(H, 10, args.n_nodes, args.tau_knbrs, self.device)
        rec_loss = torch.sum(torch.pow(data - X_, 2))
        loss_instance = instanceloss(H, CH, args.tau)
        loss_coef = torch.sum(torch.pow(Coefficient, 2))
        loss = (args.loss_instance * loss_instance) + (args.loss_knbrs * loss_knbrs) + (args.loss_coef * loss_coef) \
               + (args.rec_loss * rec_loss)

        return loss, None

    def test(self, processed_data):
        data, adj = processed_data
        self.model.eval()
        _ , _, Coefficient, _ = self.model(data, adj)
        # get C
        C = Coefficient.detach().to('cpu').numpy()
        commonZ = thrC(C, self.cfg.args.alpha)
        preds, _ = post_proC(commonZ, self.cfg.args.n_clusters)

        return preds
