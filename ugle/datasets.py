import numpy as np
import random
from typing import Union, Tuple
import torch.nn.functional as F
from ugle.logger import ugle_path, log
import torch
from karateclub.dataset import GraphReader
from torch_geometric.transforms import ToUndirected
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets.amazon import Amazon

google_store_datasets = ['acm', 'amac', 'amap', 'bat', 'citeseer', 'cora', 'cocs', 'dblp', 'eat', 'uat', 'pubmed',
                          'texas', 'wisc', 'cornell']
karate_club_datasets = ['facebook', 'twitch', 'wikipedia', 'github', 'lastfm', 'deezer']
big_datasets = ['Physics', 'CS', 'Photo', 'Computers']
all_datasets = (google_store_datasets + karate_club_datasets + big_datasets)


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



def to_edge_index(adjacency: np.ndarray):
    """
    converts adjacency in numpy array form to an array of active edges
    :param adjacency: input adjacency matrix
    :return adjacency: adjacency matrix update form
    """
    adj_label = sp.coo_matrix(adjacency)
    adj_label = adj_label.todok()

    outwards = [i[0] for i in adj_label.keys()]
    inwards = [i[1] for i in adj_label.keys()]

    adjacency = torch.tensor([outwards, inwards], dtype=int)
    return adjacency


def dropout_edge_undirected(edge_index: torch.Tensor, p: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
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


def load_real_graph_data(dataset_name: str, test_split: float = 0.5, split_scheme: str = 'drop_edges',
                         split_addition=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


    elif dataset_name in big_datasets:
        dataset_path = ugle_path + f'/data/{dataset_name}'

        if dataset_name in ['Photo', 'Computers']:
            data = Amazon(root=dataset_path, name=dataset_name, transform=ToUndirected(merge=True))[0]
        elif dataset_name in ['CS', 'Physics']:
            data = Coauthor(root=dataset_path, name=dataset_name, transform=ToUndirected(merge=True))[0]
        
        features = data.x.numpy()
        label = data.y.numpy()
        adjacency = to_dense_adj(data.edge_index).numpy().squeeze(0)


    if split_addition:
        adjacency, _ = aug_drop_adj(adjacency, drop_percent=1-split_addition, split_adj=False)

    log.debug('Splitting dataset into training/testing')
    train_adj, test_adj = split_adj(adjacency, test_split, split_scheme)

    return features, label, train_adj, test_adj

def split(n, k):
    d, r = divmod(n, k)
    return [d + 1] * r + [d] * (k - r)


def max_nodes_in_edge_index(edge_index):
    if edge_index.nelement() == 0:
        return -1
    else:
        return int(edge_index.max())

    for dataset_name in dataset_names:
        features, label, train_adjacency, test_adjacency = load_real_graph_data(dataset_name, test_split=1.)
        display_string = dataset_name + ' & '  # name
        display_string += str(train_adjacency.shape[0]) + ' & '  # n_nodes
        display_string += str(features.shape[1]) + ' & '  # n_features
        display_string += str(int(np.nonzero(train_adjacency)[0].shape[0])) + ' & '  # n_edges
        display_string += str(len(np.unique(label))) + ' & '  # n_classes
        print(display_string)
        continue
        nx_g = nx.Graph(train_adjacency)
        clustering = nx.average_clustering(nx_g)
        cercania = nx.closeness_centrality(nx_g)
        cercania = np.mean(list(cercania.values()))

    row, col = edge_index
    edge_mask = torch.rand(row.size(0)) >= p
    keep_edge_index = edge_index[:, edge_mask]
    drop_edge_index = edge_index[:, torch.ones_like(edge_mask, dtype=bool) ^ edge_mask]

        display_string += str(round(clustering, 3)) + ' & '  # clustering coefficient
        display_string += str(round(cercania, 3)) + ' \\\\'  # closeness centrality
        print(display_string)
    if visualise:
        _ = display_figure_dataset_stats(clustering_x_data, closeness_y_data, dataset_names)

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
    augmentation by randomly dropping edges with given probability
    :param input_adj: input adjacency matrix
    :param drop_percent: percent that any edge is dropped
    :return aug_adj: augmented adjacency matrix
    """

    edge_index = to_edge_index(input_adj)
    keep_index, drop_index = dropout_edge_undirected(edge_index, p=1-drop_percent)
    aug_adj = to_dense_adj(drop_index).numpy().squeeze(0)

    n_missing = input_adj.shape[0] - aug_adj.shape[0]
    pad_width = ((0, n_missing), (0, n_missing))
    aug_adj = np.pad(aug_adj, pad_width, mode='constant', constant_values=0.)
    assert input_adj.shape == aug_adj.shape

    return aug_adj, input_adj


def numpy_to_edge_index(adjacency: np.ndarray):
    """
    converts adjacency in numpy array form to an array of active edges
    :param adjacency: input adjacency matrix
    :return adjacency: adjacency matrix update form
    """
    mean, std = features.mean(), features.std()
    new_data = (features - mean) / (std + 10e-7)
    return new_data

def split_dataset(data, num_val, num_test):
    """
    splits data by dropping edges meaning no overlapping edges between sets unless 0.0 where all edges are kept
    """
    data.edge_index = remove_self_loops(data.edge_index)
    if num_test == 0.0 and num_val == 0.0:
        train_data = Data(x=data.x, y=data.y, edge_index=add_all_self_loops(data.edge_index, data.x.shape[0]))
        val_data = Data(x=data.x, y=data.y, edge_index=add_all_self_loops(data.edge_index, data.x.shape[0]))
        test_data = Data(x=data.x, y=data.y, edge_index=add_all_self_loops(data.edge_index, data.x.shape[0]))
    elif num_val == 0.0:
        train_edges, dropped_edges = dropout_edge_undirected(data.edge_index, p=num_test)
        train_data = Data(x=data.x, y=data.y, edge_index=add_all_self_loops(train_edges, data.x.shape[0]))
        test_data = Data(x=data.x, y=data.y, edge_index=add_all_self_loops(dropped_edges, data.x.shape[0]))
        val_data = Data(x=data.x, y=data.y, edge_index=add_all_self_loops(train_edges, data.x.shape[0]))
    elif num_test == 0.0:
        train_edges, dropped_edges = dropout_edge_undirected(data.edge_index, p=num_val)
        train_data = Data(x=data.x, y=data.y, edge_index=add_all_self_loops(train_edges, data.x.shape[0]))
        test_data = Data(x=data.x, y=data.y, edge_index=add_all_self_loops(train_edges, data.x.shape[0]))
        val_data = Data(x=data.x, y=data.y, edge_index=add_all_self_loops(dropped_edges, data.x.shape[0]))
    else:
        train_edges, dropped_edges = dropout_edge_undirected(data.edge_index, p=num_test+num_val)
        train_data = Data(x=data.x, y=data.y, edge_index=add_all_self_loops(train_edges, data.x.shape[0]))
        test_edges, val_edges = dropout_edge_undirected(dropped_edges, p=1-(num_test/(num_test+num_val)))
        test_data = Data(x=data.x, y=data.y, edge_index=add_all_self_loops(test_edges, data.x.shape[0]))
        val_data = Data(x=data.x, y=data.y, edge_index=add_all_self_loops(val_edges, data.x.shape[0]))
        
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
                    edge_index=to_undirected(dataset.edge_index))
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
    
    end_time = time.time() - time_spent
    log.debug(f"Time splitting: {round(end_time, 3)}s")
    return train_loader, val_loader, test_loader

def extract_dataset_stats(loader):
    temp_sampler = iter(loader)
    temp_stats = {'n_nodes': [], 'labels': [], 'n_edges': 0}
    for batch in temp_sampler:
        log.debug(f'Sampled N Nodes:  {batch.x.shape[0]}, N Features: {batch.x.shape[1]}')
        temp_stats['n_nodes'].append(batch.x.shape[0])
        temp_stats['n_features'] = batch.x.shape[1]
        temp_stats['labels'].append(np.unique(batch.y).shape[0])
        temp_stats['n_edges'] + batch.edge_index.shape[0]

    return {'n_nodes': max(temp_stats['n_nodes']), 
            'n_features': temp_stats['n_features'],
            'n_edges': temp_stats['n_edges'],
            'n_clusters': np.unique(np.array(temp_stats['labels']))[0]}
    



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
    from ugle import process 
    from torch_geometric.nn.conv import GCNConv
    from torch_geometric.nn import DMoNPooling
    from ugle.gnn_architecture import GCN
    import torch.nn as nn
    import scipy.sparse as sp

    line_profile = False
    max_nodes_per_batch = 1000000
    edges_on_gpu = True
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_cuda:
        usage = torch.cuda.mem_get_info(device)
        log.info(f'Memory GPU free/total: {usage[0]/1024/1024:.2f}MB/{usage[1]/1024/1024:.2f}MB')
        log.info(f'Memory GPU allocated: {torch.cuda.max_memory_allocated(device)/1024/1024:.2f}MB')
        log.info(f'Memory GPU reserved: {torch.cuda.max_memory_reserved(device)/1024/1024:.2f}MB\n')
    
    for dataset_name in datasets:
        print('\n')
        # how much memory does it take to load 
        mem_usage, (dataloader, val_loader, test_loader) = memory_usage((create_dataset_loader, (dataset_name, max_nodes_per_batch, 0.1, 0.2)), retval=True)
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        log.info(f"Memory CPU usage by {dataset_name}: {max(mem_usage):.2f}MB")

        # how long does it take to load data
        if line_profile:
            lp = LineProfiler()
            lp_wrapper = lp(create_dataset_loader)
            _, _, _= lp_wrapper(dataset_name, max_nodes_per_batch, 0.1, 0.2)
            lp.print_stats()


        def load_data_on_device(loader, device):
            smth = []
            for i, batch in enumerate(iter(loader)):
                log.info(i)
                if use_cuda: 
                    # GPU features and CPU edge index
                    usage = torch.cuda.mem_get_info(device)
                    log.info(f'Memory GPU free/total: {usage[0]/1024/1024:.2f}MB/{usage[1]/1024/1024:.2f}MB')
                    log.info(f'Memory GPU allocated: {torch.cuda.max_memory_allocated(device)/1024/1024:.2f}MB')

                    if edges_on_gpu: 
                        smth.append([batch.x.to(device), batch.edge_index.to(device)])
                    else: 
                        smth.append([batch.x.to(device), batch.edge_index])
                
                    usage = torch.cuda.mem_get_info(device)
                    log.info(f'Memory GPU free/total: {usage[0]/1024/1024:.2f}MB/{usage[1]/1024/1024:.2f}MB')
                    log.info(f'Memory GPU allocated: {torch.cuda.max_memory_allocated(device)/1024/1024:.2f}MB')
                    torch.cuda.reset_peak_memory_stats(device)
                

        if line_profile:
            lp = LineProfiler()
            lp_wrapper = lp(load_data_on_device)
            lp_wrapper(dataloader, device)
            lp.print_stats()
        
        #load_data_on_device(dataloader, device)

        class Net(torch.nn.Module):
            def __init__(self, dstats):
                super().__init__()

                self.layer = GCNConv(dstats['n_features'], 128, add_self_loops=False, normalize=True)
                self.pooling = DMoNPooling([128, dstats['n_clusters']], dstats['n_clusters'])

            def forward(self, x, edge_index):
                out = self.layer(x, edge_index).relu()
                adj = to_dense_adj(edge_index)
                clus_assn, out, adj, sp1, o1, c1 = self.pooling(out, adj)
                return torch.argmax(nn.functional.softmax(clus_assn.squeeze(0), dim=1), dim=1), sp1 + o1 + c1

        def forward_pass_pyg_layer(dataloader, device):
            # extract info
            dstats = extract_dataset_stats(dataloader)
            model = Net(dstats)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    return train_adjacency, validation_adjacency


if __name__ == '__main__':


    from torch_geometric.utils import add_remaining_self_loops, to_undirected, to_dense_adj
    import ugle.process as process

    def sparse_modularity(edge_index, preds, n_edges):
        degrees = torch.sparse.sum(edge_index, dim=0)._values().unsqueeze(1)
        graph_pooled = torch.spmm(torch.spmm(edge_index, preds).T, preds)
        normalizer_left = torch.spmm(preds.T, degrees)
        normalizer_right = torch.spmm(preds.T, degrees).T
        normalizer = torch.spmm(normalizer_left, normalizer_right) / 2 / n_edges
        return torch.trace(graph_pooled - normalizer) / 2 / n_edges


    def sparse_conductance(edge_index, preds):
        edge_index = edge_index.coalesce().indices()
        inter = 0
        intra = 0
        for cluster_id in np.unique(preds):
            nodes_in_cluster = torch.where(preds == cluster_id)[0]
            edges_starting_from_cluster = torch.isin(edge_index[0, :], nodes_in_cluster)
            inter_bool = torch.isin(edge_index[1, edges_starting_from_cluster], nodes_in_cluster)
            new_inter = int(torch.sum(inter_bool))
            inter += new_inter
            intra += len(inter_bool) - new_inter
        return intra / (inter + intra)
    
    def max_nodes_in_edge_index(edge_index):
        if edge_index.nelement() == 0:
            return -1
        else:
            return int(edge_index.max())
    
    def add_all_self_loops(edge_index, n_nodes):
        edge_index, _ = add_remaining_self_loops(edge_index)
        # if the end nodes have had all the edges removed then you need to manually add the final self loops
        last_in_adj = max_nodes_in_edge_index(edge_index)
        n_ids_left = torch.arange(last_in_adj + 1, n_nodes)
        edge_index = torch.concat((edge_index, torch.stack((n_ids_left, n_ids_left))), dim=1)
        return edge_index
    
    #dataset_path = ugle_path + f'/data/Computers'
    #data = Amazon(root=dataset_path, name='Computers', transform=ToUndirected(merge=True))[0]
    #edge_index = add_all_self_loops(data.edge_index, data.x.shape[0])
    #adj = to_dense_adj(edge_index).squeeze(0).numpy()
    
    for dataset in ['pubmed', 'CS', 'Physics']:
        features, label, train_adj, test_adj,  = load_real_graph_data(dataset, test_split=1.)
        edge_index = numpy_to_edge_index(train_adj)
        n_clusters = len(np.unique(label))
        n_nodes = features.shape[0]
        n_edges = edge_index.shape[1]
        n_features = features.shape[1]
        print(f'{dataset}: n_nodes:{n_nodes}, n_edges:{n_edges}, n_clusters:{n_clusters}, n_features:{n_features}')

    #assignments = torch.softmax(torch.rand((n_nodes, n_clusters)), dim=1)
    #graph = process.sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj))
    #preds = torch.argmax(assignments, dim=1)
