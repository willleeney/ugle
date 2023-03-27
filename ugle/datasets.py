import os
from omegaconf import OmegaConf, DictConfig
import numpy as np
from karateclub.dataset import GraphReader
import networkx as nx
import zipfile
import gdown
from pathlib import Path
import shutil
import torch
import copy
import random
import scipy.sparse as sp
import plotly.graph_objects as go
from typing import Union
from ugle.logger import log, ugle_path

google_store_datasets = ['acm', 'amac', 'amap', 'bat', 'citeseer', 'cora', 'cocs', 'dblp', 'eat', 'uat', 'pubmed',
                         'cite', 'corafull', 'texas', 'wisc', 'film', 'cornell']
karate_club_datasets = ['facebook', 'twitch', 'wikipedia', 'github', 'lastfm', 'deezer']
all_datasets = (google_store_datasets + karate_club_datasets)


def check_data_presence(dataset_name: str) -> bool:
    """
    checks for dataset presence in local memory
    :param dataset_name: dataset name to check
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
    downloads a graph dataset
    :param dataset_name: name of the dataset to download
    :return True if successful
    """
    log.info(f'downloading {dataset_name}')
    download_link_path = ugle_path + '/data/download_links.yaml'
    download_links = OmegaConf.load(download_link_path)
    url = download_links[dataset_name]
    dataset_path = ugle_path + f'/data/{dataset_name}'
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    dataset_zip_path = dataset_path + f'/{dataset_name}.zip'
    gdown.download(url=url, output=dataset_zip_path, quiet=False, fuzzy=True)
    log.info('finished downloading')

    # extract the zip file
    log.info('extracting dataset')
    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
        zip_ref.printdir()
        zip_ref.extractall(dataset_path)
    log.info('extraction complete')

    # correct the path dir
    extended_path = f'{dataset_path}/{dataset_name}'
    dataset_path += '/'
    if os.path.exists(extended_path):
        log.info('extraction to wrong location')
        for subdir, dirs, files in os.walk(extended_path):
            for file in files:
                extract_path = os.path.join(subdir, file)
                file = Path(file)
                out_path = os.path.join(dataset_path, file)
                log.info(f'Extracting {extract_path} to ... {out_path} ...')
                shutil.move(Path(extract_path), Path(out_path))

        shutil.rmtree(extended_path)

    return True


def load_real_graph_data(dataset_name: str, test_split: float = 0.5) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    loads the graph dataset and splits the adjacency matrix into two
    :param dataset_name: name of the dataset
    :param test_split: percentage of edges to keep
    :return features, label, train_adj, adjacency: loaded graph data
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

    elif dataset_name in karate_club_datasets:
        loader = GraphReader(dataset_name)
        features = loader.get_features().todense()
        label = loader.get_target()
        adjacency = nx.to_numpy_matrix(loader.get_graph())

    log.info('splitting dataset into training/testing')
    train_adj, test_adj = aug_drop_adj(adjacency, drop_percent=1-test_split)

    return features, label, train_adj, test_adj

def compute_datasets_info(dataset_names: list, visualise: bool=False):
    """
    computes the information about dataset statistics
    :param dataset_names: list of datasets to look at
    """

    clustering_x_data = []
    closeness_y_data = []

    for dataset_name in dataset_names:
        features, label, train_adjacency, test_adjacency = load_real_graph_data(dataset_name, test_split=1.)
        display_string = dataset_name + ' & '  # name
        display_string += str(train_adjacency.shape[0]) + ' & '  # n_nodes
        display_string += str(features.shape[1]) + ' & '  # n_features
        display_string += str(int(np.nonzero(train_adjacency)[0].shape[0]/2)) + ' & '  # n_edges
        display_string += str(len(np.unique(label))) + ' & '  # n_classes

        nx_g = nx.Graph(train_adjacency)
        clustering = nx.average_clustering(nx_g)
        cercania = nx.closeness_centrality(nx_g)
        cercania = np.mean(list(cercania.values()))

        clustering_x_data.append(clustering)
        closeness_y_data.append(cercania)

        display_string += str(round(clustering, 3)) + ' & '  # clustering coefficient
        display_string += str(round(cercania, 3)) + ' \\\\'  # closeness centrality
        if visualise:
            print(display_string)
    if visualise:
        _ = display_figure_dataset_stats(clustering_x_data, closeness_y_data, dataset_names)

    return clustering_x_data, closeness_y_data


def display_figure_dataset_stats(x_data: list, y_data: list, datasets: list):
    """
    function to display dataset statistics on a graph
    :param x_data: clustering coefficient data for x-axis
    :param y_data: closeness centrality data for y-axis
    :param datasets: list of datasets metrics were computed over
    :return fig: figure to be displayed
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_data, y=y_data, text=datasets, textposition="top center",
        color_discrete_sequence=['red']
    ))

    # layout options
    layout = dict(
                font=dict(
                    size=18),
              plot_bgcolor='white',
              paper_bgcolor='white',
              margin=dict(t=10, b=10, l=10, r=10, pad=0),
              xaxis=dict(title='Average Clustering Coefficient',
                         linecolor='black',
                         showgrid=False,
                         showticklabels=False,
                         mirror=True),
              yaxis=dict(title='Mean Closeness Centrality',
                         linecolor='black',
                         showgrid=False,
                         showticklabels=False,
                         mirror=True))
    fig.update_layout(layout)
    # save figure
    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image("images/dataset_stats.png")

    return fig


def aug_drop_features(input_feature: Union[np.ndarray, torch.Tensor], drop_percent: float = 0.2):
    """
    augmentation by randomly masking features for every node
    :param input_feature: feature matrix
    :param drop_percent: percent that any feature is dropped
    :return aug_feature: augmented feature matrix
    """
    node_num = input_feature.shape[1]
    mask_num = int(node_num * drop_percent)
    node_idx = [i for i in range(node_num)]
    mask_idx = random.sample(node_idx, mask_num)
    aug_feature = copy.deepcopy(input_feature)
    if type(aug_feature) == np.ndarray:
        zeros = np.zeros_like(aug_feature[0][0])
    else:
        zeros = torch.zeros_like(aug_feature[0][0])
    for j in mask_idx:
        aug_feature[0][j] = zeros
    return aug_feature


def aug_drop_adj(input_adj: np.ndarray, drop_percent: float = 0.2):
    """
    augmentation by randomly dropping edges with given probability
    :param input_adj: input adjacency matrix
    :param drop_percent: percent that any edge is dropped
    :return aug_adj: augmented adjacency matrix
    """

    index_list = input_adj.nonzero()
    row_idx = index_list[0].tolist()
    col_idx = index_list[1].tolist()

    index_list = []
    for i in range(len(row_idx)):
        index_list.append((row_idx[i], col_idx[i]))

    edge_num = int(len(row_idx))
    add_drop_num = int(edge_num * drop_percent)
    aug_adj = copy.deepcopy(input_adj.tolist())
    else_adj = np.zeros_like(input_adj)

    edge_idx = list(np.arange(edge_num))
    drop_idx = random.sample(edge_idx, add_drop_num)
    n_drop = len(drop_idx)
    log.info(f'dropping {n_drop} edges from {edge_num}')

    for i in drop_idx:
        aug_adj[index_list[i][0]][index_list[i][1]] = 0
        else_adj[index_list[i][0]][index_list[i][1]] = 1

    aug_adj = np.array(aug_adj)

    return aug_adj, else_adj


def numpy_to_edge_index(adjacency: np.ndarray):
    """
    converts adjacency in numpy array form to an array of active edges
    :param adjacency: input adjacency matrix
    :return adjacency: adjacency matrix update form
    """
    adj_label = sp.coo_matrix(adjacency)
    adj_label = adj_label.todok()

    outwards = [i[0] for i in adj_label.keys()]
    inwards = [i[1] for i in adj_label.keys()]

    adjacency = np.array([outwards, inwards], dtype=np.int)
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
            aug_adjacency = np.array([aug_adjacency[0], aug_adjacency[1]], dtype=np.int)
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
