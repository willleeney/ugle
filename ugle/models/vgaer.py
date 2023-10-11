# https://github.com/qcydm/VGAER/tree/main/VGAER_codes
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from fast_pytorch_kmeans import KMeans
import math
from ugle.trainer import ugleTrainer
from copy import deepcopy

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, act="tanh"):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if act == "tanh":
            self.act = nn.Tanh()
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, input):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)

        if self.bias is not None:
            output += self.bias

        return self.act(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        B_hat = z @ z.t()
        B_hat = F.sigmoid(B_hat)
        return B_hat


class VGAERModel(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim, device):
        super(VGAERModel, self).__init__()
        self.in_dim = in_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim

        layers = [GraphConvolution(self.in_dim, self.hidden1_dim, act="tanh"),
                  GraphConvolution(self.hidden1_dim, self.hidden2_dim, act="tanh"),
                  GraphConvolution(self.hidden1_dim, self.hidden2_dim, act="tanh")]
        self.layers = nn.ModuleList(layers)
        self.device = device

    def encoder(self, a_hat, features):

        h = self.layers[0](a_hat, features)
        self.mean = self.layers[1](a_hat, h)
        self.log_std = self.layers[2](a_hat, h)
        gaussian_noise = torch.randn(features.size(0), self.hidden2_dim).to(self.device)
        sampled_z = self.mean + gaussian_noise * torch.exp(self.log_std).to(self.device)
        return sampled_z

    def decoder(self, z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec

    def forward(self, a_hat, features):
        z = self.encoder(a_hat, features)
        adj_rec = self.decoder(z)
        return adj_rec, z


def compute_loss_para(adj):
    try: 
        pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        weight_mask = adj.view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0))
        weight_tensor[weight_mask] = pos_weight
    except:
        weight_tensor = torch.ones_like(adj.view(-1))
        norm = 1.
    return weight_tensor, norm


class vgaer_trainer(ugleTrainer):
    def preprocess_data(self, features, adjacency):
        A = torch.FloatTensor(adjacency)
        A[A != 0] = 1
        A_orig_ten = A.detach().clone()

        # compute B matrix
        K = 1 / (A.sum().item()) * (A.sum(dim=1).reshape(A.shape[0], 1) @ A.sum(dim=1).reshape(1, A.shape[0]))
        feats = A - K

        # compute A_hat matrix
        A = A + torch.eye(A.shape[0])
        D = torch.diag(torch.pow(A.sum(dim=1), -0.5))  # D = D^-1/2
        A_hat = D @ A @ D

        weight_tensor, norm = compute_loss_para(A)
        weight_tensor = weight_tensor
        A_hat = A_hat
        feats = feats

        return A_orig_ten, A_hat, feats, weight_tensor, norm


    def training_preprocessing(self, args, processed_data):

        self.model = VGAERModel(args.n_nodes, args.hidden1, args.hidden2, self.device).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        self.optimizers = [optimizer]

        return

    def training_epoch_iter(self, args, processed_data):
        A_orig_ten, A_hat, feats, weight_tensor, norm = processed_data
        recovered = self.model.forward(A_hat, feats)
        logits = recovered[0]
        hidemb = recovered[1]

        logits = logits.clamp(min=0., max=1.)

        loss = norm * F.binary_cross_entropy(logits.view(-1), A_orig_ten.view(-1), weight=weight_tensor)
        kl_divergence = 0.5 / logits.size(0) * (
                1 + 2 * self.model.log_std - self.model.mean ** 2 - torch.exp(self.model.log_std) ** 2).sum(
            1).mean()
        loss -= kl_divergence

        return loss, None

    def test(self, processed_data):
        A_orig_ten, A_hat, feats, weight_tensor, norm = processed_data

        self.model.eval()
        recovered = self.model.forward(A_hat, feats)
        emb = recovered[1]
        emb = emb.float().clamp(torch.finfo(torch.float32).min, torch.finfo(torch.float32).max)

        kmeans = kmeans = KMeans(n_clusters=self.cfg.args.n_clusters)
        preds = kmeans.fit_predict(emb).cpu().numpy()
      
        return preds