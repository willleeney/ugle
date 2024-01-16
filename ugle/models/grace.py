# https://github.com/CRIPAC-DIG/GRACE
import torch
import torch.nn as nn
import torch.nn.functional as F
import ugle
import scipy.sparse as sp
import numpy as np
from fast_pytorch_kmeans import KMeans
import warnings
warnings.filterwarnings('ignore')
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from torch_geometric.utils import dropout_adj

from torch_geometric.nn import GCNConv
from ugle.trainer import ugleTrainer


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x


class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

class grace_trainer(ugleTrainer):
    def preprocess_data(self, features, adjacency):
        adj_label = sp.coo_matrix(adjacency)
        adj_label = adj_label.todok()

        outwards = [i[0] for i in adj_label.keys()]
        inwards = [i[1] for i in adj_label.keys()]

        adj = torch.from_numpy(np.array([outwards, inwards], dtype=int))

        data = torch.FloatTensor(features)

        return data, adj

    def training_preprocessing(self, args, processed_data):

        activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[args.activation]
        base_model = ({'GCNConv': GCNConv})[args.base_model]

        encoder = Encoder(args.n_features, args.num_hidden, activation,
                          base_model=base_model, k=args.num_layers).to(self.device)
        self.model = Model(encoder, args.num_hidden, args.num_proj_hidden, args.tau).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        self.optimizers = [optimizer]

        return

    def training_epoch_iter(self, args, processed_data):
        data, adj = processed_data

        adj_1 = dropout_adj(adj, p=args.drop_edge_rate_1, force_undirected=True)[0]
        adj_2 = dropout_adj(adj, p=args.drop_edge_rate_2, force_undirected=True)[0]

        x_1 = ugle.datasets.aug_drop_features(data, drop_percent=args.drop_feature_rate_1)
        x_2 = ugle.datasets.aug_drop_features(data, drop_percent=args.drop_feature_rate_2)

        z1 = self.model(x_1, adj_1)
        z2 = self.model(x_2, adj_2)

        loss = self.model.loss(z1, z2, batch_size=0)

        return loss, None

    def test(self, processed_data):
        data, adj = processed_data
        self.model.eval()
        z = self.model(data, adj)
        kmeans = kmeans = KMeans(n_clusters=self.cfg.args.n_clusters)
        preds = kmeans.fit_predict(z).cpu().numpy()
        return preds



