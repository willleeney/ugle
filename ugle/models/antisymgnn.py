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
        self.centroids = None

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

            ### v3: minimise distance to cluster centroids using antisymmetric weight matrix
            if self.centroids is not None:
                conv = conv + self.center_attention @ self.centroids

            if self.bias is not None:
                conv += self.bias

            ### v1: minimise to cluster centres
            ### function that minimises distance to centriods for all points 


            ### v2: minimise distance to each other? 
            

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


        inp = self.input_dim
        self.emb = None
        if self.hidden_dim is not None:
            self.emb = Linear(self.input_dim, self.hidden_dim)
            inp = self.hidden_dim

        self.readout = Linear(inp, self.output_dim)
        self.readout_adj = Linear(inp, args.n_nodes)

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

        # dbscan + hierarchical clustering + loss
        pairwise_distances = torch.cdist(x, x, p=2)
        cluster_labels = dbscan_clustering(x, pairwise_distances, self.args.eps, self.args.minPts)
        preds, clustering_loss = merge_clusters(cluster_labels, pairwise_distances, self.args.n_clusters, linkage='complete')

        out = self.relu(self.readout(x))
        adj_hat = self.relu2(self.readout_adj(x))
        recon_feat_loss = self.recon_loss(out, features)
        recon_adj_loss = self.recon_loss(adj_hat, dense_graph)

        loss = recon_feat_loss + recon_adj_loss + (clustering_loss * self.args.alpha)
        
        if self.epoch_counter % 25 == 0:
            #kmeans = KMeans(n_clusters=self.args.n_clusters)
            #preds = kmeans.fit_predict(x.squeeze(0).detach()).cpu().numpy()

            #preds2 = dbscan_clustering(x.squeeze(0).detach(), pairwise_distances, self.args.eps, self.args.minPts)


            tsne = TSNE(n_components=2, learning_rate='auto', init='pca')
            embedding = tsne.fit_transform(x.squeeze(0).detach().cpu().numpy())
            pairwise_distances = torch.cdist(torch.from_numpy(embedding), torch.from_numpy(embedding), p=2)
            cluster_labels = dbscan_clustering(x, pairwise_distances, self.args.eps, self.args.minPts)
            preds, clustering_loss = merge_clusters(cluster_labels, pairwise_distances, self.args.n_clusters, linkage='complete')

            preds, _ = ugle.process.hungarian_algorithm(self.labels, preds.detach().cpu().numpy())
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
                   'clustering_loss': clustering_loss
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
        
        # dbscan + hierarchical clustering + loss
        pairwise_distances = torch.cdist(x, x, p=2)
        cluster_labels = dbscan_clustering(x, pairwise_distances, self.cfg.args.eps, self.cfg.args.minPts)
        preds, _ = merge_clusters(cluster_labels, pairwise_distances, self.cfg.args.n_clusters, linkage='complete')

        return preds.detach().cpu().numpy()



def contrastive_loss_no_labels(embeddings):
    """
    Compute a modified contrastive loss over a batch of embeddings where each
    point is treated as a positive example for itself and a negative example for all others.

    Args:
    - embeddings: A tensor of shape (N, F) where N is the number of points and F is the feature space.

    Returns:
    - loss: A tensor containing the contrastive loss.
    """
    # Normalize embeddings to lie on the unit sphere, which simplifies the cosine similarity calculation
    normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    # Compute the cosine similarity matrix
    similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())

    # Set the diagonal (self-similarity) to 0 because we don't want to consider it in the loss
    similarity_matrix.fill_diagonal_(0.)

    # Since we want to minimize similarity between different embeddings, the loss is the sum of all
    # positive similarities in the matrix, divided by the number of elements considered (to average it)
    loss = similarity_matrix.sum() / (similarity_matrix.numel() - similarity_matrix.size(0))
    return loss


def dbscan_clustering(node_embeddings, pairwise_distances, eps, minPts):
    # Compute the number of neighbors within the eps distance
    neighbor_counts = (pairwise_distances < eps).sum(dim=1)
    # sort those neighbours
    sorted_neighbours = torch.sort(neighbor_counts, descending=True)
    # Identify core points
    core_point_mask = sorted_neighbours[0] >= minPts
    # Compute neighbor lists for core points
    neighbor_lists = [(pairwise_distances[i] < eps).nonzero().squeeze() for i in sorted_neighbours[1][core_point_mask]]

    # Initialize with -1 (noise)
    cluster_labels = torch.full((node_embeddings.shape[0],), -1, dtype=torch.long)

    # the issue we had is that node A be in between B and C and be neighbouring both but that doesn't mean B and C are neighbours
    # so we actually want to do this but where core_point_mask is in order of most to least neighbours 
    # and every time those get assigned you want to remove those neighbours from any other nodes that may 
    # have had them from the neighbours, but actually maybe it's fine anyway because if that node was in the neighbour of the previous node
    # then it would already be assigned so that node and all of it's neighbours would've been skipped 
    # so ive got done from like 631 to 280 clusters

    # the two other ways are to not ignore this previous thing and instead just cluster all of those previous nodes that were neighbours into the same cluster
    # but the issue with this (which is why im not going to do it) is that when you have two big clusters and some points in the middle that could be either cluster
    # if you chain cluster then both clusters would end up getting clustered together, so i think it's probably just not ideal

    # also sidenote, there should be some way of determining eps and minpoints based on the density of the data...
 
    next_label = 0
    node_assigned = []
    for neighbour_idx, n_neighbours in enumerate(sorted_neighbours[0][core_point_mask]):
        node_idx = sorted_neighbours[1][core_point_mask][neighbour_idx]
        # if the current node is unassigned
        if cluster_labels[node_idx] == -1:
            # get the neighbours of the current node
            neighbors = neighbor_lists[neighbour_idx].tolist()
            # check that this node has at least 5 non assigned neighbours
            neighbors = list(set(neighbors) - set(node_assigned))
            if len(neighbors) >= minPts:
                # make all those neighbours a cluster
                cluster_labels[neighbors] = next_label
                #print(f"cluster: {next_label}  node: {node_idx}  n_neighbours: {n_neighbours}")
                next_label += 1
                node_assigned.extend(neighbors)
            

    # Assign noise points to the closest cluster
    noise_points = torch.where(cluster_labels == -1)[0]
    clustered_points =  torch.where(cluster_labels >= 0)[0]
    if noise_points.numel() > 0:
        # get the distances from noisy points to all clustered nodes
        noise_distances = pairwise_distances[noise_points, :][:, clustered_points]
        # get the closest clustered node for every noisy point 
        closest_nodes = noise_distances.argmin(dim=1)
        # assign the noisy points to the same label as their closest clustered node
        cluster_labels[noise_points] = cluster_labels[clustered_points[closest_nodes]]

    return cluster_labels


def merge_clusters(cluster_labels, pairwise_distances, k, linkage='complete'):
    """
    Merge clusters using agglomerative hierarchical clustering.

    Args:
        cluster_labels (torch.Tensor): A tensor of shape (num_nodes,) containing the initial cluster assignments.
        pairwise_distances (torch.Tensor): A tensor of shape (num_nodes, num_nodes) containing the pairwise distances.
        k (int): number of clusters needed
        linkage (str, optional): The linkage criterion to use for merging clusters ('complete', 'average', or 'single').
            Default is 'complete'.

    Returns:
        torch.Tensor: A tensor of shape (num_nodes,) containing the final cluster assignments after merging.
        torch.Tensor: A tensor containing the total intra-cluster distances for the final k clusters.
    """
    num_nodes = cluster_labels.numel()
    cluster_sizes = torch.bincount(cluster_labels)
    num_clusters = cluster_sizes.numel()

    while num_clusters > k:
        # Compute the intra-cluster distances for all cluster pairs
        intra_cluster_dists = torch.zeros((num_clusters, num_clusters))
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                cluster_i = torch.where(cluster_labels == i)[0]
                cluster_j = torch.where(cluster_labels == j)[0]
                if linkage == 'complete':
                    intra_cluster_dists[i, j] = pairwise_distances[cluster_i, :][:, cluster_j].max()
                elif linkage == 'average':
                    intra_cluster_dists[i, j] = (pairwise_distances[cluster_i, :][:, cluster_j].sum()
                                                 / (cluster_sizes[i] * cluster_sizes[j]))
                elif linkage == 'single':
                    intra_cluster_dists[i, j] = pairwise_distances[cluster_i, :][:, cluster_j].min()
                else:
                    raise ValueError(f"Invalid linkage criterion: {linkage}")

        # Find the two clusters with the smallest intra-cluster distance
        merge_indices = torch.triu_indices(num_clusters, num_clusters, offset=1)
        (i, j) = merge_indices[:, intra_cluster_dists[merge_indices[0], merge_indices[1]].argmin()]

        # Merge the two clusters
        # merge cluster labels from j into i 
        cluster_labels[cluster_labels == j] = i
        # increase the cluster size of i
        cluster_sizes[i] += cluster_sizes[j]
        # remove the cluster j from cluster sizes array
        cluster_sizes = torch.cat((cluster_sizes[:j], cluster_sizes[j + 1:]))
        # deincrement all the cluster labels bigger than j so there isn't a missing label
        cluster_labels[cluster_labels > j] -= 1

        num_clusters -= 1

    # Compute the total intra-cluster distances for the final k clusters
    total_intra_cluster_dists = torch.zeros(k)
    for i in range(len(torch.unique(cluster_labels))):
        cluster_i = torch.where(cluster_labels == i)[0]
        total_intra_cluster_dists[i] = pairwise_distances[cluster_i, :][:, cluster_i].sum() / (2 * cluster_sizes[i])

    return cluster_labels, total_intra_cluster_dists.sum()


def density_clustering_loss(pairwise_distances, eps, minPts, alpha=0.001):
    # Compute the number of neighbors within the eps distance
    neighbor_counts = (pairwise_distances < eps).sum(dim=1)
    
    # Compute the core point mask (points with >= minPts neighbors)
    core_point_mask = neighbor_counts >= minPts
    
    # Compute the distance penalty for core point pairs
    core_point_dists = pairwise_distances[core_point_mask][:, core_point_mask]
    core_point_penalty = core_point_dists[core_point_dists > eps].sum()
    
    # Compute the distance penalty for non-core point pairs
    non_core_point_dists = pairwise_distances[~core_point_mask][:, ~core_point_mask]
    non_core_point_penalty = non_core_point_dists[non_core_point_dists <= eps].sum()
    
    # Combine the penalties
    density_clustering_loss = alpha * (core_point_penalty + non_core_point_penalty)
    
    return density_clustering_loss