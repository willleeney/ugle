# https://github.com/google-research/google-research/blob/master/graph_embedding/dmon/dmon.py
import scipy.sparse as sp
from ugle.trainer import ugleTrainer
from ugle.process import preds_eval, sparse_mx_to_torch_sparse_tensor, normalize_adj
import torch.nn as nn
import torch
from ugle.gnn_architecture import GCN
import math
from collections import OrderedDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj


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


    def forward(self, features, graph_normalised, graph, extra_loss=False):

        gcn_out = self.gcn(features, graph_normalised)
        assignments = self.transform(gcn_out).squeeze(0)
        assignments = nn.functional.softmax(assignments, dim=1)

        n_edges = graph.shape[1]
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

    def embed(self, features, graph_normalised):
        gcn_out = self.gcn(features, graph_normalised)
        assignments = self.transform(gcn_out).squeeze(0)
        assignments = nn.functional.softmax(assignments, dim=1)
        return assignments



class dmon_trainer(ugleTrainer):
    def preprocess_data(self, loader):
        dataset = []
        for batch in loader:
            adjacency = to_dense_adj(batch.edge_index)
            graph = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adjacency.squeeze(0)))
            adjacency = sparse_mx_to_torch_sparse_tensor(normalize_adj(adjacency.squeeze(0)))
            features = torch.FloatTensor(batch.x)
            dataset.append(Data(x=features, y=batch.y, edge_index=adjacency, **{'graph': graph}))
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
        return dataloader

    def training_preprocessing(self, args, train_loader):
        self.model = DMoN(args).to(self.device)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.optimizers = [optimiser]
        return

    def training_epoch_iter(self, train_loader):
        for batch in train_loader:
            # transfer to device
            batch.x = batch.x.to(self.device, non_blocking=True)
            batch.edge_index = batch.edge_index.to(self.device, non_blocking=True)
            #graph = torch.sparse_coo_tensor(indices=batch.graph, values=torch.ones(batch.graph.shape[1]))._coalesced_(True)
            graph = graph.to(self.device, non_blocking=True)
            # forward and backward pass
            loss = self.model(batch.x, batch.edge_index, graph)
            self.optimizers[0].zero_grad()
            loss.backward()
            self.optimizers[0].step()
            # right now this is only appropriate for a single batch
            break
        
        return loss

    def test(self, test_loader, eval_metrics):
        multi_batch_metric_info = None
        with torch.no_grad():
            for batch in test_loader:
                batch.x, batch.edge_index = batch.x.to(self.device, non_blocking=True), batch.edge_index.to(self.device, non_blocking=True)
                assignments = self.model.embed(batch.x, batch.edge_index).detach().cpu()
                #graph = torch.sparse_coo_tensor(indices=batch.graph, values=torch.ones(batch.graph.shape[1]))._coalesced_(True)
                results, eval_preds = preds_eval(batch.y, assignments, batch.graph, metrics=eval_metrics, sf=4)
                # right now this is only appropriate for a single batch
                break                                          

        return results









