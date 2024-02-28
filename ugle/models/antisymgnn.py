import ugle
import scipy.sparse as sp
from ugle.trainer import ugleTrainer
import torch.nn as nn
import torch
import math
import wandb
from fast_pytorch_kmeans import KMeans
import plotly.graph_objs as go
from sklearn.manifold import TSNE
import numpy as np

from torch.nn import Parameter, init, Linear, ModuleList
from torch_geometric.nn import MessagePassing, GCNConv
from typing import Optional


class AntiSymmetricConv(MessagePassing):
    def __init__(self, 
                 in_channels: int,
                 num_iters: int = 1, 
                 gamma: float = 0.1, 
                 epsilon : float = 0.1, 
                 activ_fun: str = 'tanh', # it should be monotonically non-decreasing
                 gcn_conv: bool = False,
                 bias: bool = True,
                 train_weights: bool = True) -> None:

        super().__init__(aggr = 'add')
        self.train_weights = train_weights
        self.W = Parameter(torch.empty((in_channels, in_channels)), requires_grad=self.train_weights)
        self.bias = Parameter(torch.empty(in_channels), requires_grad=self.train_weights) if bias else None

        self.lin = Linear(in_channels, in_channels, bias=False) # for simple aggregation
        if not self.train_weights:
            self.lin.weight.requires_grad = False
        self.I = Parameter(torch.eye(in_channels), requires_grad=False)

        self.gcn_conv = GCNConv(in_channels, in_channels, bias=False) if gcn_conv else None
        if not self.train_weights and self.gcn_conv is not None:
            for param in self.gcn_conv.parameters():
                param.requires_grad = False

        self.num_iters = num_iters
        self.gamma = gamma
        self.epsilon = epsilon
        self.activation = getattr(torch, activ_fun)

        self.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        antisymmetric_W = self.W - self.W.T - self.gamma * self.I

        for _ in range(self.num_iters):
            if self.gcn_conv is None:
                # simple aggregation
                neigh_x = self.lin(x) 
                neigh_x = self.propagate(edge_index, x=neigh_x, edge_weight=edge_weight)
            else:
                # gcn aggregation
                neigh_x = self.gcn_conv(x, edge_index=edge_index, edge_weight=edge_weight)

            conv = x @ antisymmetric_W.T + neigh_x

            if self.bias is not None:
                conv += self.bias

            x = x + self.epsilon * self.activation(conv)
        return x

    def message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def reset_parameters(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.lin.reset_parameters()
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)


class antisymgnn(nn.Module):
    def __init__(self,
                 args,
                 act='tanh'):
        """Initializes the layer with specified parameters."""
        super(antisymgnn, self).__init__()
        self.args = args
        self.n_clusters = args.n_clusters

        self.input_dim = args.n_features
        self.output_dim = args.n_features
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.epsilon = args.epsilon
        self.gamma = args.gamma
        self.activ_fun = act
        self.bias = True
        self.train_weights = True
        self.weight_sharing = True

        self.recon_loss = nn.MSELoss()
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()


        inp = self.input_dim
        self.emb = None
        if self.hidden_dim is not None:
            self.emb = Linear(self.input_dim, self.hidden_dim)
            inp = self.hidden_dim

        self.readout = Linear(inp, self.output_dim)
        self.readout_adj = Linear(inp, args.n_nodes)
        self.readout_assignments = Linear(inp, args.n_clusters)

        self.conv = ModuleList()
        if self.weight_sharing:
            self.conv.append(
                AntiSymmetricConv(in_channels = inp,
                                  num_iters = self.num_layers,
                                  gamma = self.gamma,
                                  epsilon = self.epsilon,
                                  activ_fun = self.activ_fun,
                                  gcn_conv = args.gcn_norm,
                                  bias = self.bias,
                                  train_weights=self.train_weights)
            )
        else:
            for _ in range(args.num_layers):
                self.conv.append(
                    AntiSymmetricConv(in_channels = inp,
                                      num_iters = 1,
                                      gamma = self.gamma,
                                      epsilon = self.epsilon,
                                      activ_fun = self.activ_fun,
                                      gcn_conv = args.gcn_norm,
                                      bias = self.bias,
                                      train_weights = self.train_weights)
                )

        self.epoch_counter = 0 
        wandb.init(project='antisymgnn', entity='phd-keep-learning')
        return


    def forward(self, features, graph, dense_graph):

        x = self.emb(features) if self.emb else features
        for conv in self.conv:
            x = conv(x, graph)
        out = self.relu(self.readout(x))
        adj_hat = self.relu2(self.readout_adj(x))
        assignments = self.relu3(self.readout_assignments(x)).squeeze(0)
        assignments = nn.functional.softmax(assignments, dim=1)

        recon_feat_loss = self.recon_loss(out, features)
        recon_adj_loss = self.recon_loss(adj_hat, dense_graph)

        loss = recon_feat_loss + recon_adj_loss

        if self.epoch_counter > 1000:
            adj = torch.sparse.FloatTensor(graph, torch.ones_like(graph[0,:], dtype=torch.float32), (self.args.n_nodes, self.args.n_nodes)).to(graph.device)
            n_edges = adj._nnz()
            degrees = torch.sparse.sum(adj, dim=0)._values().unsqueeze(1).to(torch.float32)
            #degrees = torch.bincount(graph[0, :])
            #assert torch.bincount(graph[0, :]).shape[0] == self.args.nodes

            graph_pooled = torch.spmm(torch.spmm(adj, assignments).T, assignments)
            normalizer_left = torch.spmm(assignments.T, degrees)
            normalizer_right = torch.spmm(assignments.T, degrees).T
            normalizer = torch.spmm(normalizer_left, normalizer_right) / 2 / n_edges
            spectral_loss = - torch.trace(graph_pooled - normalizer) / 2 / n_edges
            loss += spectral_loss
            wandb.log({'spectral_loss': spectral_loss}, commit=False)
        
        if self.epoch_counter % 25 == 0:
            kmeans = KMeans(n_clusters=self.cfg.args.n_clusters)
            preds = kmeans.fit_predict(x.squeeze(0).detach()).cpu().numpy()

            tsne = TSNE(n_components=2, learning_rate='auto', init='pca')
            embedding = tsne.fit_transform(x.squeeze(0).detach().cpu().numpy())

            preds, _ = ugle.process.hungarian_algorithm(self.labels, preds)
            border_colors = ['green' if pred == label else 'red' for pred, label in zip(preds, self.labels)]
        
            fig = go.Figure(data=go.Scatter(x=embedding[:, 0], y=embedding[:, 1], mode='markers',
                                                marker=dict(
                                                        color=self.labels,  # Node color
                                                        line=dict(
                                                            color=border_colors,  # Border color
                                                            width=1),
                                                        size=5,
                                                        colorscale='Spectral',
                                                        )
                                                    ))
            acc = round(float(sum(preds == self.labels) /len(self.labels)), 3)

            wandb.log({"tsne_vis": wandb.Plotly(fig),
                        "acc": acc}, commit=False)

        wandb.log({'loss': loss,
                   'recon_feat_loss': recon_feat_loss,
                   'recon_adj_loss': recon_adj_loss,
                   }, commit=True)
        
        self.epoch_counter += 1
        return loss

    def embed(self, features, graph):
        x = self.emb(features) if self.emb else features
        for conv in self.conv:
            x = conv(x, graph)
        return x
        #assignments = self.relu3(self.readout_assignments(x)).squeeze(0)
        #return nn.functional.softmax(assignments, dim=1).squeeze(0).detach().cpu().numpy().argmax(axis=1)


class antisymgnn_trainer(ugleTrainer):

    def preprocess_data(self, features, adjacency):
        features = torch.FloatTensor(features)
        features[features != 0.] = 1.

        adjacency = adjacency + sp.eye(adjacency.shape[0])
        adjacency = sp.coo_matrix(adjacency)

        adj_label = adjacency.todok()

        outwards = [i[0] for i in adj_label.keys()]
        inwards = [i[1] for i in adj_label.keys()]

        adj = torch.from_numpy(np.array([outwards, inwards], dtype=int))

        return features, adj, torch.Tensor(adjacency.todense())

    def training_preprocessing(self, args, processed_data):

        self.model = antisymgnn(args).to(self.device)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.optimizers = [optimiser]
        return

    def training_epoch_iter(self, args, processed_data):
        loss = self.model(*processed_data)
        return loss, None

    def test(self, processed_data):
        features, graph, _ = processed_data
        with torch.no_grad():
            x = self.model.embed(features, graph)
        
        kmeans = KMeans(n_clusters=self.cfg.args.n_clusters)
        preds = kmeans.fit_predict(x.squeeze(0).detach()).cpu().numpy()

        return preds
