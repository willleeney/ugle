import numpy as np
import scipy.sparse as sp
import torch
from scipy.linalg import fractional_matrix_power, inv
from sklearn.metrics import f1_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
import math
from line_profiler import LineProfiler
from typing import Tuple

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf


def compute_ppr(adj: np.ndarray, alpha: int = 0.2, self_loop: bool = True) -> float:
    """
    Computes personalized PageRank diffusion.

    Args:
        adj (np.ndarray): Adjacency matrix
        alpha (float): Teleport probability (default: 0.2)
        self_loop (bool): Whether to add self-loops (default: True)

    Returns:
        float: Personalized PageRank matrix
    """
    if self_loop:
        adj = adj + np.eye(adj.shape[0])  # A^ = A + I_n
    d = np.diag(np.sum(adj, 1))  # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)  # D^(-1/2)
    at = np.matmul(np.matmul(dinv, adj), dinv)  # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(adj.shape[0]) - (1 - alpha) * at))  # a(I_n-(1-a)A~)^-1


def modularity(adjacency: np.ndarray, preds: np.ndarray) -> float:
    """
    Computes modularity.

    Args:
        adjacency (np.ndarray): Adjacency matrix
        preds (np.ndarray): Predicted cluster assignments

    Returns:
        float: Modularity score
    """
    adjacency = sp.coo_matrix(adjacency).tocsr()
    degrees = adjacency.sum(axis=0).A1
    m = degrees.sum()
    result = 0
    for cluster_id in np.unique(preds):
        cluster_indices = np.where(preds == cluster_id)[0]
        adj_submatrix = adjacency[cluster_indices, :][:, cluster_indices]
        degrees_submatrix = degrees[cluster_indices]
        result += np.sum(adj_submatrix) - (np.sum(degrees_submatrix) ** 2) / m
    return result / m


def conductance(adjacency: np.ndarray, preds: np.ndarray) -> float:
    """
    Computes conductance.

    Args:
        adjacency (np.ndarray): Adjacency matrix
        preds (np.ndarray): Predicted cluster assignments

    Returns:
        float: Conductance score
    """
    if len(np.unique(preds)) == 1:
        return 1.
    inter = 0
    intra = 0
    cluster_idx = np.zeros(adjacency.shape[0], dtype=bool)
    for cluster_id in np.unique(preds):
        cluster_idx[:] = 0
        cluster_idx[np.where(preds == cluster_id)[0]] = 1
        adj_submatrix = adjacency[cluster_idx, :]
        inter += np.sum(adj_submatrix[:, cluster_idx])
        intra += np.sum(adj_submatrix[:, ~cluster_idx])
    return intra / (inter + intra)


def preds_eval(labels: np.ndarray, preds: np.ndarray, sf=4, adj: np.ndarray = None, 
               metrics=['nmi', 'f1']) -> Tuple[dict, np.ndarray]:
    """
    Evaluates predictions given metrics.

    Args:
        labels (np.ndarray): True labels
        preds (np.ndarray): Predicted labels
        sf (int): Significant figures for rounding (default: 4)
        adj (np.ndarray): Adjacency matrix (default: None)
        metrics (list): List of metrics to compute (default: ['nmi', 'f1'])

    Returns:
        results (dict): Results dictionary 
        eval_preds (np.ndarray): Evaluated predictions
    """
    # returns the correct predictions to match labels
    eval_preds, _ = hungarian_algorithm(labels, preds)
    results = {}

    if 'nmi' in metrics:
        nmi = normalized_mutual_info_score(labels, eval_preds)
        results['nmi'] = float(round(nmi, sf))

    if 'f1' in metrics:
        f1 = f1_score(labels, eval_preds, average='macro')
        results['f1'] = float(round(f1, sf))

    if 'snmi' in metrics:
        true_num_clusters = len(np.unique(labels))
        found_num_clusters = len(np.unique(preds))

        scaling_k = np.exp(-(np.abs(true_num_clusters - found_num_clusters) / true_num_clusters))
        snmi = scaling_k * nmi
        results['snmi'] = float(round(snmi, sf))

    if 'modularity' in metrics:
        assert adj is not None, 'adj not provided'
        results['modularity'] = float(round(modularity(adj, eval_preds), sf))
    if 'conductance' in metrics:
        assert adj is not None, 'adj not provided'
        results['conductance'] = float(round(conductance(adj, eval_preds), sf))

    if 'n_clusters' in metrics:
        results['n_clusters'] = len(np.unique(preds))

    return results, eval_preds


def hungarian_algorithm(labels: np.ndarray, preds: np.ndarray, col_ind=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hungarian algorithm for prediction reassignment.

    Args:
        labels (np.ndarray): True labels
        preds (np.ndarray): Predicted labels
        col_ind (np.ndarray): Column indices for reassignment (default: None)

    Returns:
        preds (np.ndarray): Reassigned predictions 
        col_ind (np.ndarray): Column indices for reassigning future predictions
    """
    labels = labels.astype(int)
    assert preds.size == labels.size
    D = max(preds.max(), labels.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(preds.size):
        w[preds[i], labels[i]] += 1

    if col_ind is None:
        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        preds = col_ind[preds]

    else:
        preds = [col_ind[int(i)] for i in preds]
        preds = np.asarray(preds)

    return preds, col_ind


def preprocess_features(features: np.ndarray):
    """
    Row-normalize feature matrix and convert to tuple representation.

    Args:
        features: Input feature matrix

    Returns:
        np.ndarray: Preprocessed feature matrix
    """
    rowsum = np.array(features.sum(1))
    nonzero_indexes = np.argwhere(rowsum != 0).flatten()
    r_inv = np.zeros_like(rowsum, dtype=float)
    r_inv[nonzero_indexes] = np.power(rowsum[nonzero_indexes], -1)
    r_inv[np.isinf(r_inv)] = 0.
    r_inv[np.isnan(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(features)


def normalize_adj(adj: np.ndarray):
    """
    Symmetrically normalize adjacency matrix.

    Args:
        adj (np.ndarray): Input adjacency matrix

    Returns:
        sp.coo_matrix: Normalized adjacency matrix
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.csr_matrix) -> torch.FloatTensor:
    """
    Convert a scipy sparse matrix to a torch sparse tensor.

    Args:
        sparse_mx (sp.csr_matrix): Input sparse matrix

    Returns:
        torch.FloatTensor: Torch sparse tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(int))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def euclidean_distance(point1: np.ndarray, point2: np.ndarray):
    """
    Calculate the Euclidean distance between two points.

    Args:
        point1 (np.ndarray): First point
        point2 (np.ndarray): Second point

    Returns:
        float: Euclidean distance between the two points
    """
    point_iterable = zip(point1, point2)
    distance = math.sqrt(sum([(y - x) ** 2 for x, y in point_iterable]))
    return distance