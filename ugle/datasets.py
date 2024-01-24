import os
from omegaconf import OmegaConf, DictConfig
import numpy as np
import networkx as nx
import zipfile
import gdown
from pathlib import Path
import shutil
import torch
import copy
import random
import scipy.sparse as sp
from typing import Union, Tuple
from ugle.logger import log, ugle_path
from torch_geometric.utils import to_dense_adj, stochastic_blockmodel_graph
import torch
from torch_geometric.transforms import ToUndirected
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets.amazon import Amazon

google_store_datasets = ['acm', 'amac', 'amap', 'bat', 'citeseer', 'cora', 'cocs', 'dblp', 'eat', 'uat', 'pubmed',
                          'texas', 'wisc', 'cornell']
big_datasets = ['Physics', 'CS', 'Photo', 'Computers']
all_datasets = (google_store_datasets + big_datasets)


def check_data_presence(dataset_name: str) -> bool:
    """
    Checks for dataset presence in local memory.

    Args:
        dataset_name (str): Dataset name to check

    Returns:
        bool: True if dataset is present, False otherwise
    """
    dataset_path = ugle_path + f'/data/{dataset_name}'
    if not os.path.exists(dataset_path):
        return False
    elif not os.path.exists(f'{dataset_path}/{dataset_name}_feat.npy'):
        return False
    elif not os.path.exists(f'{dataset_path}/{dataset_name}_label.npy'):
        return False
    elif not os.path.exists(f'{dataset_path}/{dataset_name}_adj.npy'):
        return False
    else:
        return True


def download_graph_data(dataset_name: str) -> bool:
    """
    Downloads a graph dataset.

    Args:
        dataset_name (str): Name of the dataset to download

    Returns:
        bool: True if successful, False otherwise
    """
    log.info(f'Downloading {dataset_name}')
    download_link_path = ugle_path + '/data/download_links.yaml'
    download_links = OmegaConf.load(download_link_path)
    url = download_links[dataset_name]
    dataset_path = ugle_path + f'/data/{dataset_name}'
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    dataset_zip_path = dataset_path + f'/{dataset_name}.zip'
    gdown.download(url=url, output=dataset_zip_path, quiet=False, fuzzy=True)
    log.info('Finished downloading')

    # extract the zip file
    log.info('Extracting dataset')
    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
        zip_ref.printdir()
        zip_ref.extractall(dataset_path)
    log.info('Extraction complete')

    # correct the path dir
    extended_path = f'{dataset_path}/{dataset_name}'
    dataset_path += '/'
    if os.path.exists(extended_path):
        log.info('Extraction to wrong location')
        for subdir, dirs, files in os.walk(extended_path):
            for file in files:
                extract_path = os.path.join(subdir, file)
                file = Path(file)
                out_path = os.path.join(dataset_path, file)
                log.info(f'Extracting {extract_path} to ... {out_path} ...')
                shutil.move(Path(extract_path), Path(out_path))

        shutil.rmtree(extended_path)

    return True



def to_edge_index(adjacency: np.ndarray) -> torch.Tensor:
    """
    Converts adjacency in numpy array form to an array of active edges.

    Args:
        adjacency (np.ndarray): Input adjacency matrix

    Returns:
        torch.Tensor: Edge index tensor
    """
    adj_label = sp.coo_matrix(adjacency)
    adj_label = adj_label.todok()

    outwards = [i[0] for i in adj_label.keys()]
    inwards = [i[1] for i in adj_label.keys()]

    adjacency = torch.tensor([outwards, inwards], dtype=int)
    return adjacency


def dropout_edge_undirected(edge_index: torch.Tensor, p: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies dropout to undirected edges in the edge index.

    Args:
        edge_index (torch.Tensor): Edge index tensor
        p (float): Dropout probability (default: 0.5)

    Returns:
        keep_edge_index (torch.Tensor): Tensor for kept edges
        drop_edge_index (torch.Tensor): Tensor for dropped edges
    """
    if p <= 0. or p >= 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 -- (got {p})')
    
    edge_index = edge_index[:, torch.where(edge_index[1, :] > edge_index[0, :])[0]]

    row, col = edge_index
    edge_mask = torch.rand(row.size(0)) >= p
    keep_edge_index = edge_index[:, edge_mask]
    drop_edge_index = edge_index[:, torch.ones_like(edge_mask, dtype=bool) ^ edge_mask]

    keep_edge_index = torch.cat([keep_edge_index, keep_edge_index.flip(0)], dim=1)
    drop_edge_index = torch.cat([drop_edge_index, drop_edge_index.flip(0)], dim=1)

    return keep_edge_index, drop_edge_index


def load_real_graph_data(dataset_name: str, test_split: float = 0.5, 
                         split_scheme: str = 'drop_edges') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the graph dataset and splits the adjacency matrix into two.

    Args:
        dataset_name (str): Name of the dataset
        test_split (float): Percentage of edges to keep for testing (default: 0.5)
        split_scheme (str): Splitting scheme for the adjacency matrix (default: 'drop_edges')
    
    Returns:
        features (np.ndarray): Feature matrix of the graph
        label (np.ndarray): Labelled ground-truth of the graph
        train_adj (np.ndarray): Training adjacency matrix
        test_adj (np.ndarray): Testing adjacency matrix
    
    
    """
    assert dataset_name in all_datasets, f"{dataset_name} not a real dataset"

    if dataset_name in google_store_datasets:
        if not check_data_presence(dataset_name):
            extraction_success = download_graph_data(dataset_name)
            assert extraction_success, f'download/extraction of dataset {dataset_name} failed'
            assert check_data_presence(dataset_name)

        dataset_path = ugle_path + f'/data/{dataset_name}/{dataset_name}'
        features = np.load(dataset_path + "_feat.npy", allow_pickle=True)
        label = np.load(dataset_path + "_label.npy", allow_pickle=True)
        adjacency = np.load(dataset_path + "_adj.npy", allow_pickle=True)

    elif dataset_name in big_datasets:
        dataset_path = ugle_path + f'/data/{dataset_name}'

        if dataset_name in ['Photo', 'Computers']:
            data = Amazon(root=dataset_path, name=dataset_name, transform=ToUndirected(merge=True))[0]
        elif dataset_name in ['CS', 'Physics']:
            data = Coauthor(root=dataset_path, name=dataset_name, transform=ToUndirected(merge=True))[0]
        
        features = data.x.numpy()
        label = data.y.numpy()
        adjacency = to_dense_adj(data.edge_index).numpy().squeeze(0)

    log.debug('Splitting dataset into training/testing')
    train_adj, test_adj = split_adj(adjacency, test_split, split_scheme)

    return features, label, train_adj, test_adj


def compute_datasets_info(dataset_names: list, 
                          compute_stats: bool=False) -> Union[float, float]:
    """
    Computes the information about dataset statistics.

    Args:
        dataset_names (list): List of datasets to analyze
        compute_stats (bool): Whether to compute additional statistics (default: False)

    Returns:
        clustering_x_data (float): Clustering Coefficient
        closeness_y_data (float): Closeness Centralilty 
    """
    clustering_x_data = []
    closeness_y_data = []

    for dataset_name in dataset_names:
        features, label, train_adjacency, test_adjacency = load_real_graph_data(dataset_name, test_split=1.)
        display_string = dataset_name + ' & '  # name
        display_string += str(train_adjacency.shape[0]) + ' & '  # n_nodes
        display_string += str(features.shape[1]) + ' & '  # n_features
        display_string += str(int(np.nonzero(train_adjacency)[0].shape[0])) + ' & '  # n_edges
        display_string += str(len(np.unique(label))) + ' & '  # n_classes
        if compute_stats: 
            nx_g = nx.Graph(train_adjacency)
            clustering = nx.average_clustering(nx_g)
            cercania = nx.closeness_centrality(nx_g)
            cercania = np.mean(list(cercania.values()))

            clustering_x_data.append(clustering)
            closeness_y_data.append(cercania)

            display_string += str(round(clustering, 3)) + ' & '  # clustering coefficient
            display_string += str(round(cercania, 3)) + ' \\\\'  # closeness centrality
        print(display_string)
   
    return clustering_x_data, closeness_y_data


def aug_drop_features(input_feature: Union[np.ndarray, torch.Tensor], 
                      drop_percent: float = 0.2) -> Union[np.ndarray, torch.Tensor]:
    """
    Augmentation by randomly masking features for every node.

    Args:
        input_feature (Union[np.ndarray, torch.Tensor]): Feature matrix
        drop_percent (float): Percent that any feature is dropped (default: 0.2)

    Returns:
        aug_feature (Union[np.ndarray, torch.Tensor]): Augmented feature matrix
    """
    node_num = input_feature.shape[1]
    mask_num = int(node_num * drop_percent)
    node_idx = [i for i in range(node_num)]
    mask_idx = random.sample(node_idx, mask_num)
    aug_feature = input_feature.detach().clone()
    if type(aug_feature) == np.ndarray:
        zeros = np.zeros_like(aug_feature[0][0])
    else:
        zeros = torch.zeros_like(aug_feature[0][0])
    for j in mask_idx:
        aug_feature[0][j] = zeros
    return aug_feature


def aug_drop_adj(input_adj: np.ndarray, drop_percent: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augmentation by randomly dropping edges with given probability from a dense matrix.

    Args:
        input_adj (np.ndarray): Input adjacency matrix
        drop_percent (float): Percent that any edge is dropped (default: 0.2)

    Returns:
        aug_adj (np.ndarray): Augmented adjacency matrix
        input_adj (np.ndarray): Original adjacency matrix
    """

    edge_index = to_edge_index(input_adj)
    keep_index, drop_index = dropout_edge_undirected(edge_index, p=1-drop_percent)
    aug_adj = to_dense_adj(drop_index).numpy().squeeze(0)

    n_missing = input_adj.shape[0] - aug_adj.shape[0]
    pad_width = ((0, n_missing), (0, n_missing))
    aug_adj = np.pad(aug_adj, pad_width, mode='constant', constant_values=0.)
    assert input_adj.shape == aug_adj.shape

    return aug_adj, input_adj


def numpy_to_edge_index(adjacency: np.ndarray) -> np.ndarray:
    """
    Converts adjacency in numpy array form to an array of active edges.

    Args:
        adjacency (np.ndarray): Input adjacency matrix

    Returns:
        np.ndarray: Edge index array
    """
    adj_label = sp.coo_matrix(adjacency)
    adj_label = adj_label.todok()

    outwards = [i[0] for i in adj_label.keys()]
    inwards = [i[1] for i in adj_label.keys()]

    adjacency = np.array([outwards, inwards], dtype=int)
    return adjacency


class Augmentations:
    """
    A utility for graph data augmentation
    """

    def __init__(self, method='gdc'):
        methods = {"split", "standardize", "katz"}
        assert method in methods
        self.method = method

    @staticmethod
    def _split(features, permute=True):
        """
        Data augmentation is build by spliting data along the feature dimension.

        :param data: the data object to be augmented
        :param permute: Whether to permute along the feature dimension

        """
        perm = np.random.permutation(features.shape[1]) if permute else np.arange(features.shape[1])
        features = features[:, perm]
        size = features.shape[1] // 2
        x1 = features[:, :size]
        x2 = features[:, size:]

        return x1, x2

    @staticmethod
    def _standardize(features):
        """
        Applies a zscore node feature data augmentation.

        :param data: The data to be augmented
        :return: a new augmented instance of the input data
        """
        mean, std = features.mean(), features.std()
        new_data = (features - mean) / (std + 10e-7)
        return new_data

    @staticmethod
    def _katz(features, adjacency, beta=0.1, threshold=0.0001):
        """
        Applies a Katz-index graph topology augmentation

        :param data: The data to be augmented
        :return: a new augmented instance of the input data
        """
        n_nodes = features.shape[0]

        a_hat = adjacency + sp.eye(n_nodes)
        d_hat = sp.diags(
            np.array(1 / np.sqrt(a_hat.sum(axis=1))).reshape(n_nodes))
        a_hat = d_hat @ a_hat @ d_hat
        temp = sp.eye(n_nodes) - beta * a_hat
        h_katz = (sp.linalg.inv(temp.tocsc()) * beta * a_hat).toarray()
        mask = (h_katz < threshold)
        h_katz[mask] = 0.
        edge_index = np.array(h_katz.nonzero())
        edge_attr = torch.tensor(h_katz[h_katz.nonzero()], dtype=torch.float32)

        return edge_index

    def __call__(self, features, adjacency):
        """
        Applies different data augmentation techniques
        """

        if self.method == "katz":
            aug_adjacency = self._katz(features, adjacency)
            aug_adjacency = np.array([aug_adjacency[0], aug_adjacency[1]], dtype=int)
            adjacency = adjacency.todense()
            adjacency = numpy_to_edge_index(adjacency)
            aug_features = features.copy()
        elif self.method == 'split':
            features, aug_features = self._split(features)
            adjacency = adjacency.todense()
            adjacency = numpy_to_edge_index(adjacency)
            aug_adjacency = adjacency.copy()
        elif self.method == "standardize":
            aug_features = self._standardize(features)
            adjacency = adjacency.todense()
            adjacency = numpy_to_edge_index(adjacency)
            aug_adjacency = adjacency.copy()

        return features, adjacency, aug_features, aug_adjacency

    def __str__(self):
        return self.method.title()


def split(n: int, k: int) -> list:
    """
    From a number of nodes and number of clusters, splits the nodes so that each cluster has an interger
    number of nodes.

    Args: 
        n (int): number of nodes
        k (int): number of clusters

    Returns: 
        list: cluster sizes for each cluster
    """
    d, r = divmod(n, k)
    return [d + 1] * r + [d] * (k - r)


def create_synth_graph(n_nodes: int, n_features: int , n_clusters: int, adj_type: str, 
                       feature_type: str = 'random') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a synthetic graph with specified properties.

    Args:
        n_nodes (int): Number of nodes in the graph
        n_features (int): Number of features for each node
        n_clusters (int): Number of clusters in the graph
        adj_type (str): Type of adjacency matrix ('disjoint', 'random', 'complete')
        feature_type (str): Type of node features ('random', 'complete', 'disjoint') (default: 'random')

    Returns:
        adj (np.ndarray): generated adjacency matrix
        features (np.ndarray): node features
        labels (np.ndarray): ground-truth
    """
    if adj_type == 'disjoint':
        probs = (np.identity(n_clusters)).tolist()
    elif adj_type == 'random':
        probs = (np.ones((n_clusters, n_clusters))/n_clusters).tolist()
    elif adj_type == 'complete':
        probs = np.ones((n_clusters, n_clusters)).tolist()

    cluster_sizes = split(n_nodes, n_clusters)
    adj = to_dense_adj(stochastic_blockmodel_graph(cluster_sizes, probs)).squeeze(0).numpy()
    
    if feature_type == 'random':
        features = torch.normal(mean=0, std=1, size=(n_nodes, n_features)).numpy()
        features = np.where(features > 0., 1, 0)
    elif feature_type == 'complete': 
        features = np.ones((n_nodes, n_features))
    elif feature_type == 'disjoint':
        features = np.zeros((n_nodes, n_features))
        feature_dims_fo_cluster = split(n_features, n_clusters)
        start_feat = 0
        end_feat = feature_dims_fo_cluster[0]
        start_clus = 0
        end_clus = cluster_sizes[0]
        for i in range(len(feature_dims_fo_cluster)):
            features[start_clus:end_clus, start_feat:end_feat] = np.ones_like(features[start_clus:end_clus, start_feat:end_feat])
            if i == len(feature_dims_fo_cluster) - 1:
                break
            start_feat += feature_dims_fo_cluster[i]
            end_feat += feature_dims_fo_cluster[i+1]
            start_clus += cluster_sizes[i]
            end_clus += cluster_sizes[i+1]

    labels = []
    for i in range(n_clusters):
        labels.extend([i] * cluster_sizes[i])
    labels = np.array(labels)

    return adj, features.astype(float), labels


def split_adj(adj: np.ndarray, percent: float, split_scheme: str):
    """
    Splits the adjacency matrix based on the specified split scheme and percentage.

    Args:
        adj (np.ndarray): Input adjacency matrix
        percent (float): Percentage of edges to keep
        split_scheme (str): Splitting scheme ('drop_edges', 'split_edges', 'all_edges', 'no_edges')

    Returns:
        train_adjacency (np.ndarray): Training adjacency matrix
        validation_adjacency (np.ndarray): Validation adjacency matrix
    """
    if split_scheme == 'drop_edges':
        # drops edges from dataset to form new adj 
        if percent != 1.:
            train_adjacency, validation_adjacency = aug_drop_adj(adj, drop_percent=1 - percent)
        else:
            train_adjacency = adj
            validation_adjacency = adj.copy()
    elif split_scheme == 'split_edges':
        # splits the adj via the edges so that no edges in both 
        if percent != 1.:
            train_adjacency, validation_adjacency = aug_drop_adj(adj, drop_percent=1 - percent)
        else:
            train_adjacency = adj
            validation_adjacency = adj.copy()

    elif split_scheme == 'all_edges':
        # makes the adj fully connected 
        train_adjacency = np.ones_like(adj)
        validation_adjacency = np.ones_like(adj)

    elif split_scheme == 'no_edges':
        # makes the adj completely unconnected 
        train_adjacency = np.zeros_like(adj)
        validation_adjacency = np.zeros_like(adj)
    
    return train_adjacency, validation_adjacency