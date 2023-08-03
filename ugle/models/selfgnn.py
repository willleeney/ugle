# https://github.com/zekarias-tilahun/SelfGNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from fast_pytorch_kmeans import KMeans
import scipy.sparse as sp
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from functools import wraps
import copy
import ugle
from ugle.trainer import ugleTrainer


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class Normalize(nn.Module):
    def __init__(self, dim=None, method="batch"):
        super().__init__()
        method = None if dim is None else method
        if method == "batch":
            self.norm = nn.BatchNorm1d(dim)
        elif method == "layer":
            self.norm = nn.LayerNorm(dim)
        else:  # No norm => identity
            self.norm = lambda x: x

    def forward(self, x):
        return self.norm(x)


class Encoder(nn.Module):

    def __init__(self, args, layers, heads, dropout=None):
        super().__init__()
        rep_dim = layers[-1]
        self.gnn_type = args.gnn_type
        self.dropout = dropout
        self.project = args.prj_head_norm

        self.stacked_gnn = get_encoder(args.gnn_type, layers, heads, args.concat)
        self.encoder_norm = Normalize(dim=rep_dim, method=args.encoder_norm)
        if self.project != "no":
            self.projection_head = nn.Sequential(
                nn.Linear(rep_dim, rep_dim),
                Normalize(dim=rep_dim, method=args.prj_head_norm),
                nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(self, x, edge_index, edge_weight=None):
        for i, gnn in enumerate(self.stacked_gnn):
            if self.gnn_type == "gat" or self.gnn_type == "sage":
                x = gnn(x, edge_index)
            else:
                x = gnn(x, edge_index, edge_weight=edge_weight)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.encoder_norm(x)
        return x, (self.projection_head(x) if self.project != "no" else None)


class SelfGNN(nn.Module):

    def __init__(self, args, layers, heads, dropout=0.0, moving_average_decay=0.99):
        super().__init__()
        self.student_encoder = Encoder(args, layers=layers, heads=heads, dropout=dropout)
        self.teacher_encoder = None
        self.teacher_ema_updater = EMA(moving_average_decay)
        rep_dim = layers[-1]
        self.student_predictor = nn.Sequential(
            nn.Linear(rep_dim, rep_dim),
            Normalize(dim=rep_dim, method=args.prd_head_norm),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout))

    @singleton('teacher_encoder')
    def _get_teacher_encoder(self):
        teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(teacher_encoder, False)
        return teacher_encoder

    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

    def encode(self, x, edge_index, edge_weight=None, encoder=None):
        encoder = self.student_encoder if encoder is None else encoder
        encoder.train(self.training)
        return encoder(x, edge_index, edge_weight)

    def forward(self, x1, x2, edge_index_v1, edge_index_v2, edge_weight_v1=None, edge_weight_v2=None):
        """
        Apply student network on both views

        v<x>_rep is the output of the stacked GNN
        v<x>_student is the output of the student projection head, if used, otherwise is just a reference to v<x>_rep
        """
        v1_enc = self.encode(x=x1, edge_index=edge_index_v1, edge_weight=edge_weight_v1)
        v1_rep, v1_student = v1_enc if v1_enc[1] is not None else (v1_enc[0], v1_enc[0])
        v2_enc = self.encode(x=x2, edge_index=edge_index_v2, edge_weight=edge_weight_v2)
        v2_rep, v2_student = v2_enc if v2_enc[1] is not None else (v2_enc[0], v2_enc[0])

        """
        Apply the student predictor both views using the outputs from the previous phase 
        (after the stacked GNN or projection head - if there is one)
        """
        v1_pred = self.student_predictor(v1_student)
        v2_pred = self.student_predictor(v2_student)

        """
        Apply the same procedure on the teacher network as in the student network except the predictor.
        """
        with torch.no_grad():
            teacher_encoder = self._get_teacher_encoder()
            v1_enc = self.encode(x=x1, edge_index=edge_index_v1, edge_weight=edge_weight_v1, encoder=teacher_encoder)
            v1_teacher = v1_enc[1] if v1_enc[1] is not None else v1_enc[0]
            v2_enc = self.encode(x=x2, edge_index=edge_index_v2, edge_weight=edge_weight_v2, encoder=teacher_encoder)
            v2_teacher = v2_enc[1] if v2_enc[1] is not None else v2_enc[0]

        """
        Compute symmetric loss (once based on view1 (v1) as input to the student and then using view2 (v2))
        """
        loss1 = loss_fn(v1_pred, v2_teacher.detach())
        loss2 = loss_fn(v2_pred, v1_teacher.detach())

        loss = loss1 + loss2
        return v1_rep, v2_rep, loss.mean()


def get_encoder(gnn_type, layers, heads, concat):
    """
    Builds the GNN backbone as required
    """
    if gnn_type == "gcn":
        return nn.ModuleList([GCNConv(layers[i - 1], layers[i]) for i in range(1, len(layers))])
    elif gnn_type == "sage":
        return nn.ModuleList([SAGEConv(layers[i - 1], layers[i]) for i in range(1, len(layers))])
    elif gnn_type == "gat":
        return nn.ModuleList(
            [GATConv(layers[i - 1], layers[i] // heads[i - 1], heads=heads[i - 1], concat=concat)
             for i in range(1, len(layers))])


class selfgnn_trainer(ugleTrainer):
    def preprocess_data(self, features, adjacency):

        adjacency = sp.csr_matrix(adjacency)

        augmentation = ugle.datasets.Augmentations(method=self.cfg.args.aug)
        features, adjacency, aug_features, aug_adjacency = augmentation(features, adjacency)

        features = torch.FloatTensor(features)
        adjacency = torch.LongTensor(adjacency)
        aug_features = torch.FloatTensor(aug_features)
        aug_adjacency = torch.LongTensor(aug_adjacency)

        diff = abs(aug_features.shape[1] - features.shape[1])
        if diff > 0:
            """
            Data augmentation on the features could lead to mismatch between the shape of the two views,
            hence the smaller view should be padded with zero. (smaller_data is a reference, changes will
            reflect on the original data)
            """
            which_small = 'features' if features.shape[1] < aug_features.shape[1] else 'aug_features'
            smaller_data = features if which_small == 'features' else aug_features
            smaller_data = F.pad(smaller_data, pad=(0, diff))
            if which_small == 'features':
                features = smaller_data
            else:
                aug_features = smaller_data
            features = F.normalize(features)
            aug_features = F.normalize(aug_features)
            self.cfg.args.n_features = features.shape[1]

        return features, adjacency, aug_features, aug_adjacency

    def training_preprocessing(self, args, processed_data):

        layers = [args.n_features, args.layer1, args.layer2]
        heads = [args.head1, args.head2]
        self.model = SelfGNN(args=args, layers=layers, heads=heads).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.optimizers = [optimizer]

        return

    def training_epoch_iter(self, args, processed_data):
        features, adjacency, aug_features, aug_adjacency = processed_data
        v1_output, v2_output, loss = self.model(
            x1=features, x2=aug_features, edge_index_v1=adjacency, edge_index_v2=aug_adjacency)
        self.model.update_moving_average()
        return loss, None

    def test(self, processed_data):
        features, adjacency, aug_features, aug_adjacency = processed_data

        self.model.eval()

        v1_output, v2_output, _ = self.model(
            x1=features, x2=aug_features, edge_index_v1=adjacency,
            edge_index_v2=aug_adjacency)

        emb = torch.cat([v1_output, v2_output], dim=1).detach()
        emb = emb.squeeze(0)
        kmeans = kmeans = KMeans(n_clusters=self.cfg.args.n_clusters)
        preds = kmeans.fit_predict(emb).cpu().numpy()

        return preds

