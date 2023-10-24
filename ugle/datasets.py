import numpy as np
import random
from typing import Union, Tuple
import torch.nn.functional as F
from ugle.logger import ugle_path, log
import torch
import time
from torch_geometric.datasets import WikiCS, Reddit2, Planetoid, Coauthor, Flickr, WebKB, AttributedGraphDataset, NELL, GitHub
from torch_geometric.datasets.amazon import Amazon
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.utils import add_remaining_self_loops, to_undirected, to_dense_adj, stochastic_blockmodel_graph
from torch_geometric.transforms import ToUndirected, NormalizeFeatures, Compose



def split(n, k):
    d, r = divmod(n, k)
    return [d + 1] * r + [d] * (k - r)


def max_nodes_in_edge_index(edge_index):
    if edge_index.nelement() == 0:
        return -1
    else:
        return int(edge_index.max())


def dropout_edge_undirected(edge_index: torch.Tensor, p: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    if p <= 0. or p >= 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 -- (got {p})')

    row, col = edge_index
    edge_mask = torch.rand(row.size(0)) >= p
    keep_edge_index = edge_index[:, edge_mask]
    drop_edge_index = edge_index[:, torch.ones_like(edge_mask, dtype=bool) ^ edge_mask]

    return keep_edge_index, drop_edge_index


def add_all_self_loops(edge_index, n_nodes):
    edge_index, _ = add_remaining_self_loops(edge_index)
    # if the end nodes have had all the edges removed then you need to manually add the final self loops
    last_in_adj = max_nodes_in_edge_index(edge_index)
    n_ids_left = torch.arange(last_in_adj + 1, n_nodes)
    edge_index = torch.concat((edge_index, torch.stack((n_ids_left, n_ids_left))), dim=1)
    return edge_index


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
    aug_feature = input_feature.detach().clone()
    if type(aug_feature) == np.ndarray:
        zeros = np.zeros_like(aug_feature[0][0])
    else:
        zeros = torch.zeros_like(aug_feature[0][0])
    for j in mask_idx:
        aug_feature[0][j] = zeros
    return aug_feature


def standardize(features):
    """
    Applies a zscore node feature data augmentation.

    :param data: The data to be augmented
    :return: a new augmented instance of the input data
    """
    mean, std = features.mean(), features.std()
    new_data = (features - mean) / (std + 10e-7)
    return new_data

def split_dataset(data, num_val, num_test):
    # split data into train/val/test
    train_edges, dropped_edges = dropout_edge_undirected(data.edge_index, p=num_test+num_val)
    train_data = Data(x=data.x, y=data.y, edge_index=add_all_self_loops(train_edges, data.x.shape[0]))
    test_edges, val_edges = dropout_edge_undirected(dropped_edges, p=1-(num_test/(num_test+num_val)))
    val_data = Data(x=data.x, y=data.y, edge_index=add_all_self_loops(val_edges, data.x.shape[0]))
    test_data = Data(x=data.x, y=data.y, edge_index=add_all_self_loops(test_edges, data.x.shape[0]))
    return train_data, val_data, test_data


def create_dataset_loader(dataset_name, max_batch_nodes, num_val, num_test):
    # load dataset
    dataset_path = ugle_path + f'/data/{dataset_name}'
    undir_transform = Compose([ToUndirected(merge=True), NormalizeFeatures()])
    data = None
    start = time.time()
    if dataset_name == 'WikiCS':
        data = WikiCS(root=dataset_path, is_undirected=True, transform=undir_transform)[0]
    elif dataset_name == 'Reddit':
        data = Reddit2(root=dataset_path, transform=undir_transform)[0]
    elif dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        data = Planetoid(root=dataset_path, name=dataset_name, transform=undir_transform)[0]
    elif dataset_name in ['Photo', 'Computers']:
        data = Amazon(root=dataset_path, name=dataset_name, transform=undir_transform)[0]
    elif dataset_name in ['CS', 'Physics']:
        data = Coauthor(root=dataset_path, name=dataset_name, transform=undir_transform)[0]
    elif dataset_name == 'Flickr':
        data = Flickr(root=dataset_path, transform=undir_transform)[0]
    elif dataset_name in ['Facebook', 'PPI', 'Wiki']:
        data = AttributedGraphDataset(root=dataset_path, name=dataset_name, transform=undir_transform)[0]
    elif dataset_name in ['Texas', 'Cornell', 'Wisconsin']:
        data = WebKB(root=dataset_path, name=dataset_name, transform=undir_transform)[0]
    elif dataset_name == 'GitHub':
        data = GitHub(root=dataset_path, transform=undir_transform)[0]
    elif dataset_name == 'NELL':
        dataset = NELL(root=dataset_path)[0]
        data = Data(x=F.normalize(dataset.x.to_dense(), dim=0), y=dataset.y, 
                    edge_index=to_undirected(add_all_self_loops(dataset.edge_index, dataset.x.size(dim=0))))
    else:
        raise NameError(f'{dataset_name} is not a valid dataset_name parameter')

    time_spent = time.time()
    log.info(f"Time loading {dataset_name}: {round(time_spent - start, 3)}s")
    log.info(f'Full N Nodes:  {data.x.shape[0]}, N Features: {data.x.shape[1]}')

    # split dataset
    train_data, val_data, test_data = split_dataset(data, num_val, num_test)

    # create samplers 
    if data.num_nodes > max_batch_nodes:
        train_loader = NeighborLoader(
            train_data,
            num_neighbors=[10, 10],
            batch_size=128,
            directed=False
        )
        val_loader = NeighborLoader(
            val_data,
            num_neighbors=[10, 10],
            batch_size=128,
            directed=False
        )
        test_loader = NeighborLoader(
            test_data,
            num_neighbors=[10, 10],
            batch_size=128,
            directed=False
        )
    else: 
        train_loader = DataLoader([train_data], batch_size=1, shuffle=False, num_workers=6)
        val_loader = DataLoader([val_data], batch_size=1, shuffle=False, num_workers=6)
        test_loader = DataLoader([test_data], batch_size=1, shuffle=False, num_workers=6)
    
    sampled_data = next(iter(train_loader))
    log.debug(f'Sampled N Nodes:  {sampled_data.x.shape[0]}, N Features: {sampled_data.x.shape[1]}')
    end_time = time.time() - time_spent
    log.debug(f"Time splitting: {round(end_time, 3)}s")
    return train_loader, val_loader, test_loader


def create_synth_graph(n_nodes: int, n_features: int , n_clusters: int, num_val: float, num_test: float, adj_type: str = 'random', feature_type: str = 'random'):
    if adj_type == 'disjoint':
        probs = (np.identity(n_clusters)).tolist()
    elif adj_type == 'random':
        probs = (np.ones((n_clusters, n_clusters))/n_clusters).tolist()
    elif adj_type == 'complete':
        probs = np.ones((n_clusters, n_clusters)).tolist()

    cluster_sizes = split(n_nodes, n_clusters)
    adj = stochastic_blockmodel_graph(cluster_sizes, probs)
    
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
    labels = torch.Tensor(labels)

    data = Data(x=torch.Tensor(features), y=labels, edge_index=adj)
    train_data, val_data, test_data = split_dataset(data, num_val, num_test)

    train_loader = DataLoader([train_data], batch_size=1, shuffle=False, num_workers=6)
    val_loader = DataLoader([val_data], batch_size=1, shuffle=False, num_workers=6)
    test_loader = DataLoader([test_data], batch_size=1, shuffle=False, num_workers=6)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    datasets = ['Texas', 'Wisconsin', 'Cornell', 'Cora', 'CiteSeer', 'Photo',
        'Computers', 'CS', 'PubMed', 'Physics', 'Flickr', 'Facebook', 'PPI',
        'Wiki', 'WikiCS', 'NELL', 'GitHub', 'Reddit']
    from memory_profiler import memory_usage
    from line_profiler import LineProfiler

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        usage = torch.cuda.mem_get_info(device)
        log.info(f'Memory usage: {usage[0]/1024/1024/1024:.2f}GB/{usage[1]/1024/1024/1024:.2f}GB')
        log.info(f'Memory GPU allocated: {torch.cuda.max_memory_allocated(device)/1024/1024/1024:.2f}GB')
        log.info(f'Memory GPU reserved: {torch.cuda.max_memory_reserved(device)/1024/1024/1024:.2f}GB')

    
    for dataset_name in ['Flickr']:
        train_loader, val_loader, test_loader = create_dataset_loader(dataset_name, 100000, 0.1, 0.2)

        # how much memory does it take to load 
        mem_usage = memory_usage((create_dataset_loader, (dataset_name, 1000, 0.1, 0.2)))
        log.info(f"Max memory usage by {dataset_name}: {max(mem_usage):.2f}MB")

        # how long does it take to load data
        lp = LineProfiler()
        lp_wrapper = lp(create_dataset_loader)
        _, _, _= lp_wrapper(dataset_name, 1000, 0.1, 0.2)
        lp.print_stats()

        # load as is and use the data preprocessing 
        from ugle import process 
        import scipy as sp 

        def pre_process_dgi(loader):
            dataset = []
            for batch in loader:
                adjacency = to_dense_adj(batch.edge_index)
                adjacency = adjacency.squeeze(0) + np.eye(adjacency.shape[1])
                adjacency = process.normalize_adj(adjacency)
                adj = process.sparse_mx_to_torch_sparse_tensor(adjacency)
                features = process.preprocess_features(batch.x)
                features = torch.FloatTensor(features[np.newaxis])
                dataset.append(Data(x=features, y=batch.y, adj=adj))
            
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            return dataloader

        #lp = LineProfiler()
        #lp_wrapper = lp(pre_process_dgi)
        #dataloader = lp_wrapper(iter(train_loader))
        #lp.print_stats()

        def load_data_on_device(loader, device):
            smth = []
            for i, batch in enumerate(iter(loader)):
                log.info(i)
                if use_cuda: 
                    # GPU features and CPU edge index
                    usage = torch.cuda.mem_get_info(device)
                    log.info(f'Memory usage: {usage[0]/1024/1024/1024:.2f}GB/{usage[1]/1024/1024/1024:.2f}GB')
                    log.info(f'Memory GPU allocated: {torch.cuda.max_memory_allocated(device)/1024/1024/1024:.2f}GB')
                    log.info(f'Memory GPU reserved: {torch.cuda.max_memory_reserved(device)/1024/1024/1024:.2f}GB')

                    smth.append([batch.x.to(device),  batch.y, batch.edge_index.to(device)])
                    usage = torch.cuda.mem_get_info(device)
                    log.info(f'Memory usage: {usage[0]/1024/1024/1024:.2f}GB/{usage[1]/1024/1024/1024:.2f}GB')
                    log.info(f'Memory GPU allocated: {torch.cuda.max_memory_allocated(device)/1024/1024/1024:.2f}GB')
                    log.info(f'Memory GPU reserved: {torch.cuda.max_memory_reserved(device)/1024/1024/1024:.2f}GB')


        #lp = LineProfiler()
        #lp_wrapper = lp(load_data_on_device)
        #lp_wrapper(train_loader, device)
        #lp.print_stats()

        load_data_on_device(train_loader, device)

