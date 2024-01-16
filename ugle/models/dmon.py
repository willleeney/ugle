# https://github.com/google-research/google-research/blob/master/graph_embedding/dmon/dmon.py
import ugle
import scipy.sparse as sp
from ugle.trainer import ugleTrainer
import torch.nn as nn
import torch
from ugle.gnn_architecture import GCN
import math
from collections import OrderedDict

class DMoN(nn.Module):
    def __init__(self,
                 args,
                 act='selu',
                 do_unpooling=False):
        """Initializes the layer with specified parameters."""
        super(DMoN, self).__init__()
        self.args = args
        self.n_clusters = args.n_clusters
        self.orthogonality_regularization = args.orthogonality_regularization
        self.cluster_size_regularization = args.cluster_size_regularization
        self.dropout_rate = args.dropout_rate
        self.do_unpooling = do_unpooling
        self.gcn = GCN(args.n_features, args.architecture, act=act, skip=True)
        self.transform = nn.Sequential(OrderedDict([
            ('layer1', nn.Linear(args.architecture, args.n_clusters)),
            ('dropout', nn.Dropout(args.dropout_rate)),
        ]))

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data, gain=math.sqrt(2))
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

        self.transform.apply(init_weights)

        return


    def forward(self, graph, graph_normalised, features, extra_loss=False):

        gcn_out = self.gcn(features, graph_normalised, sparse=True, skip=True)

        assignments = self.transform(gcn_out).squeeze(0)
        assignments = nn.functional.softmax(assignments, dim=1)

        n_edges = graph._nnz()
        degrees = torch.sparse.sum(graph, dim=0)._values().unsqueeze(1)
        graph_pooled = torch.spmm(torch.spmm(graph, assignments).T, assignments)
        normalizer_left = torch.spmm(assignments.T, degrees)
        normalizer_right = torch.spmm(assignments.T, degrees).T
        normalizer = torch.spmm(normalizer_left, normalizer_right) / 2 / n_edges
        spectral_loss = - torch.trace(graph_pooled - normalizer) / 2 / n_edges
        loss = spectral_loss

        if extra_loss:
            pairwise = torch.spmm(assignments.T, assignments)
            identity = torch.eye(self.n_clusters).to(pairwise.device)
            orthogonality_loss = torch.norm(pairwise / torch.norm(pairwise) -
                                         identity / math.sqrt(float(self.n_clusters)))
            orthogonality_loss *= self.orthogonality_regularization
            loss += orthogonality_loss

            cluster_loss = torch.norm(torch.sum(pairwise, dim=1)) / self.args.n_nodes * math.sqrt(float(self.n_clusters)) - 1
            cluster_loss *= self.cluster_size_regularization
            loss += cluster_loss

        else:
            cluster_sizes = torch.sum(assignments, dim=0)
            cluster_loss = torch.norm(cluster_sizes) / self.args.n_nodes * math.sqrt(float(self.n_clusters)) - 1
            cluster_loss *= self.cluster_size_regularization
            loss += cluster_loss

        return loss

    def embed(self, graph, graph_normalised, features):
        gcn_out = self.gcn(features, graph_normalised, sparse=True)
        assignments = self.transform(gcn_out).squeeze(0)
        assignments = nn.functional.softmax(assignments, dim=1)

        return assignments


class dmon_trainer(ugleTrainer):

    def preprocess_data(self, features, adjacency):
        features = torch.FloatTensor(features)
        features[features != 0.] = 1.


        adjacency = adjacency + sp.eye(adjacency.shape[0])
        adj_label = adjacency.copy()
        adjacency = ugle.process.normalize_adj(adjacency)

        graph_normalised = ugle.process.sparse_mx_to_torch_sparse_tensor(adjacency)

        adj_label = sp.coo_matrix(adj_label)
        graph = ugle.process.sparse_mx_to_torch_sparse_tensor(adj_label)

        return graph, graph_normalised, features

    def training_preprocessing(self, args, processed_data):

        self.model = DMoN(args).to(self.device)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.optimizers = [optimiser]

        return

    def training_epoch_iter(self, args, processed_data):
        graph, graph_normalised, features = processed_data
        loss = self.model(graph, graph_normalised, features)
        
        return loss, None

    def test(self, processed_data):
        graph, graph_normalised, features = processed_data
        with torch.no_grad():
            assignments = self.model.embed(graph, graph_normalised, features)
            preds = assignments.detach().cpu().numpy().argmax(axis=1)

        return preds




