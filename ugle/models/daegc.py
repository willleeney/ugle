# code insipred from https://github.com/Tiger101010/DAEGC
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from fast_pytorch_kmeans import KMeans
import scipy.sparse as sp
import ugle
from ugle.logger import log
from ugle.trainer import ugleTrainer
from ugle.gnn_architecture import GAT

class DAEGC(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha, num_clusters, v=1):
        super(DAEGC, self).__init__()
        self.num_clusters = num_clusters
        self.v = v

        # get pretrain model
        self.gat = GAT(num_features, hidden_size, embedding_size, alpha)
        #self.gat.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x, adj, M):
        A_pred, z = self.gat(x, adj, M)
        q = self.get_Q(z)

        return A_pred, z, q

    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def get_M(adj):
    adj_numpy = adj.todense()
    # t_order
    t=2
    row_l1_norms = np.linalg.norm(adj_numpy, ord=1, axis=1)
    tran_prob = adj_numpy / row_l1_norms[:, None]
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)


class daegc_trainer(ugleTrainer):

    def preprocess_data(self, features, adjacency):
        adjacency = adjacency + sp.eye(adjacency.shape[0])
        adj_label = adjacency.copy()
        adjacency = ugle.process.normalize_adj(adjacency)
        M = get_M(adjacency)
        adj = torch.FloatTensor(adjacency.todense())
        adj_label = torch.FloatTensor(adj_label)
        features = torch.FloatTensor(features)
        features[features != 0.] = 1.
        return features, adj, adj_label, M

    def training_preprocessing(self, args, processed_data):
        features, adj, adj_label, M = processed_data

        log.debug('creating model')
        self.model = DAEGC(num_features=args.n_features, hidden_size=args.hidden_size,
                      embedding_size=args.embedding_size, alpha=args.alpha, num_clusters=args.n_clusters).to(
            self.device)
        optimizer = Adam(self.model.gat.parameters(), lr=args.pre_lr, weight_decay=args.weight_decay)

        log.debug('pretraining')
        best_nmi = 0.
        for pre_epoch in range(args.pre_epoch):
            # training pass
            self.model.train()
            A_pred, z = self.model.gat(features, adj, M)
            loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        log.debug('kmeans init estimate')
        with torch.no_grad():
            _, z = self.model.gat(features, adj, M)

        kmeans = kmeans = KMeans(n_clusters=self.cfg.args.n_clusters)
        _ = kmeans.fit_predict(z).cpu().numpy()
        
        self.model.cluster_layer.data = kmeans.centroids.clone().detach().to(self.device)

        log.debug('model training')
        optimizer = Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        self.optimizers = [optimizer]
        return


    def training_epoch_iter(self, args, processed_data):
        if len(processed_data) == 4:
            features, adj, adj_label, M = processed_data
        else:
            features, adj, adj_label, M, Q = processed_data


        if self.current_epoch % args.update_interval == 0:
            # update_interval
            A_pred, z, Q = self.model(features, adj, M)
            q = Q.detach().data.cpu().numpy().argmax(1)

        A_pred, z, q = self.model(features, adj, M)
        p = target_distribution(Q.detach())

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        re_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))

        loss = (args.kl_loss_const * kl_loss) + re_loss

        processed_data = (features, adj, adj_label, M, Q)

        return loss, processed_data

    def test(self, processed_data):

        with torch.no_grad():
            features, adj, adj_label, M = processed_data
            _, z, Q = self.model(features, adj, M)
            preds = Q.detach().data.cpu().numpy().argmax(1)

        return preds