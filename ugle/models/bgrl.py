# https://github.com/Namkyeong/BGRL_Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fast_pytorch_kmeans import KMeans
import scipy.sparse as sp
from torch_geometric.nn import GCNConv
import copy
import warnings
warnings.filterwarnings('ignore')
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from torch_geometric.utils import dropout_adj
import ugle
from ugle.trainer import ugleTrainer

class EMA:
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        beta = 1 - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        return old * beta + (1 - beta) * new


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class Encoder(nn.Module):

    def __init__(self, layer_config, dropout=None, project=False, **kwargs):
        super().__init__()

        self.conv1 = GCNConv(layer_config[0], layer_config[1])
        self.bn1 = nn.BatchNorm1d(layer_config[1], momentum=0.01)
        self.prelu1 = nn.PReLU()
        self.conv2 = GCNConv(layer_config[1], layer_config[2])
        self.bn2 = nn.BatchNorm1d(layer_config[2], momentum=0.01)
        self.prelu2 = nn.PReLU()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.prelu1(self.bn1(x))
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.prelu2(self.bn2(x))

        return x


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class BGRL(nn.Module):

    def __init__(self, layer_config, pred_hid, dropout=0.0, moving_average_decay=0.99, epochs=1000, **kwargs):
        super().__init__()
        self.student_encoder = Encoder(layer_config=layer_config, dropout=dropout, **kwargs)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(self.teacher_encoder, False)
        self.teacher_ema_updater = EMA(moving_average_decay, epochs)
        rep_dim = layer_config[-1]
        self.student_predictor = nn.Sequential(nn.Linear(rep_dim, pred_hid), nn.PReLU(), nn.Linear(pred_hid, rep_dim))
        self.student_predictor.apply(init_weights)

    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

    def forward(self, x1, x2, edge_index_v1, edge_index_v2, edge_weight_v1=None, edge_weight_v2=None):
        v1_student = self.student_encoder(x=x1, edge_index=edge_index_v1, edge_weight=edge_weight_v1)
        v2_student = self.student_encoder(x=x2, edge_index=edge_index_v2, edge_weight=edge_weight_v2)

        v1_pred = self.student_predictor(v1_student)
        v2_pred = self.student_predictor(v2_student)

        with torch.no_grad():
            v1_teacher = self.teacher_encoder(x=x1, edge_index=edge_index_v1, edge_weight=edge_weight_v1)
            v2_teacher = self.teacher_encoder(x=x2, edge_index=edge_index_v2, edge_weight=edge_weight_v2)

        loss1 = loss_fn(v1_pred, v2_teacher.detach())
        loss2 = loss_fn(v2_pred, v1_teacher.detach())

        loss = loss1 + loss2
        return v1_student, v2_student, loss.mean()


class bgrl_trainer(ugleTrainer):
    def preprocess_data(self, features, adjacency):
        adj_label = sp.coo_matrix(adjacency)
        adj_label = adj_label.todok()

        outwards = [i[0] for i in adj_label.keys()]
        inwards = [i[1] for i in adj_label.keys()]

        adj = torch.from_numpy(np.array([outwards, inwards], dtype=int))

        data = torch.FloatTensor(features)

        return data, adj

    def training_preprocessing(self, args, processed_data):

        layers = [args.n_features, args.layer1, args.layer2]
        self.model = BGRL(layer_config=layers, pred_hid=args.pred_hidden,
                     dropout=args.dropout, epochs=args.max_epoch).to(self.device)

        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        if args.max_epoch == 1000:
            args.max_epoch += 1
        scheduler = lambda epoch: epoch / 1000 if epoch < 1000 else (1 + np.cos((epoch - 1000) * np.pi / (args.max_epoch - 1000))) * 0.5

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)

        self.scheduler = scheduler
        self.optimizers = [optimizer]

        return

    def training_epoch_iter(self, args, processed_data):

        data, adj = processed_data
        self.model.update_moving_average()
        adj_1 = dropout_adj(adj, p=args.drop_edge_rate_1, force_undirected=True)[0]
        adj_2 = dropout_adj(adj, p=args.drop_edge_rate_2, force_undirected=True)[0]

        x_1 = ugle.datasets.aug_drop_features(data, drop_percent=args.drop_feature_rate_1)
        x_2 = ugle.datasets.aug_drop_features(data, drop_percent=args.drop_feature_rate_2)

        v1_output, v2_output, loss = self.model(
            x1=x_1, x2=x_2, edge_index_v1=adj_1, edge_index_v2=adj_2,
            edge_weight_v1=None, edge_weight_v2=None)

        return loss, None

    def test(self, processed_data):
        data, adj = processed_data
        self.model.eval()
        v1_output, v2_output, _ = self.model(
                x1=data, x2=data, edge_index_v1=adj, edge_index_v2=adj,
            edge_weight_v1=None,
            edge_weight_v2=None)

        kmeans = kmeans = KMeans(n_clusters=self.cfg.args.n_clusters)
        preds = kmeans.fit_predict(v1_output).cpu().numpy()


        return preds