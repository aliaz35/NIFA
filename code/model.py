import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import copy
import downstream

from utils import fair_matrix

class VictimModel():
    def __init__(self, in_feats, h_feats, num_classes, device, name='GCN', task_type='NC'):
        
        assert name in ['GCN', 'SGC', 'APPNP', 'GraphSAGE', 'GAT', 'GCNII', 'GIN'], "GNN model not implement"
        if name == 'GCN':
            self.model = GCN(in_feats, h_feats, h_feats)
        elif name == 'SGC':
            self.model = SGC(in_feats, h_feats, num_classes)
        elif name == 'APPNP':
            self.model = APPNP(in_feats, h_feats, num_classes)
        elif name == 'GraphSAGE':
            self.model = GraphSAGE(in_feats, h_feats, num_classes)
        elif name == "GAT":
            self.model = GAT(in_feats, h_feats)
        elif name == "GCNII":
            self.model = GCNII(in_feats, h_feats)
        elif name == "GIN":
            self.model = GIN(in_feats, h_feats)

        self.model.to(device)

        self.task = downstream.Classifier(h_feats, num_classes, device)

    def optimize(self, g, index_split, epochs, lr, patience):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=0)
        loss_fn = nn.CrossEntropyLoss()

        self.model.train()
        feature = g.ndata['feature']
        label = g.ndata['label']

        train_index, val_index, test_index = index_split['train_index'], index_split['val_index'], index_split['test_index']

        best_val_acc = 0
        cnt = 0
        for epoch in range(epochs):
            output = self.task(self.model(g, feature))
            pred = output.argmax(1)
            val_acc = torch.eq(pred, label)[val_index].sum() / len(val_index)
            test_acc = torch.eq(pred, label)[test_index].sum() / len(test_index)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state_dict = copy.deepcopy(self.model.state_dict())
                cnt = 0
            else:
                cnt += 1        
            if cnt >= patience and epoch > 200:
                break 

            loss = loss_fn(output[train_index], label[train_index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.model.load_state_dict(best_model_state_dict)

    def re_optimize(self, g, uncertainty, index_split, epochs, lr, patience, defense):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=0)
        loss_fn = nn.CrossEntropyLoss()

        self.model.train()
        feature = g.ndata['feature']
        label = g.ndata['label']

        # import pdb; pdb.set_trace()
        mask = torch.zeros(uncertainty.shape[0]).to(feature.device)
        mask[index_split['train_index']] = 1

        unc = torch.where(mask==1, uncertainty, 1)
        _, train_idx = torch.sort(unc, descending=False)
        train_index = train_idx[:int((1-defense)*index_split['train_index'].shape[0])]

        val_index, test_index = index_split['val_index'], index_split['test_index']

        best_val_acc = 0
        cnt = 0
        for epoch in range(epochs):
            output = self.task(self.model(g, feature))
            pred = output.argmax(1)
            val_acc = torch.eq(pred, label)[val_index].sum() / len(val_index)
            test_acc = torch.eq(pred, label)[test_index].sum() / len(test_index)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state_dict = copy.deepcopy(self.model.state_dict())
                cnt = 0
            else:
                cnt += 1        
            if cnt >= patience and epoch > 200:
                break 

            loss = loss_fn(output[train_index], label[train_index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.model.load_state_dict(best_model_state_dict)

    def eval(self, g, index_split):
        with torch.no_grad():
            self.model.eval()
            self.task.eval()
            output = self.task(self.model(g, g.ndata['feature']))
            label = g.ndata['label']
            sensitive = g.ndata['sensitive']
            pred = output.argmax(1)

            test_index = index_split['test_index']
            test_acc = torch.eq(pred, label)[test_index].sum().item() / len(test_index)

            SP, EO = fair_matrix(pred, label, sensitive, test_index)
            sp = torch.tensor(SP).abs().mean().item()
            eo = torch.tensor(EO).abs().mean().item()

            return test_acc, sp, eo


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, h_feats, norm='both', activation=F.relu)
        self.conv2 = dgl.nn.GraphConv(h_feats, h_feats, norm='both', activation=F.relu)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        # h = F.relu(h)
        h = self.conv2(g, h)
        return h

class GAT(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int=2):
        super().__init__()
        self.gat1 = dgl.nn.GATConv(in_dim,                 hidden_dim, num_heads=num_heads, activation=F.relu)
        self.gat2 = dgl.nn.GATConv(hidden_dim * num_heads, hidden_dim, num_heads=1, activation=F.relu)

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        h = self.gat1(g, feat)
        h = h.view(h.shape[0], -1)
        h = self.gat2(g, h)
        return h.squeeze(1)

class GCNII(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.gcnii1 = dgl.nn.GCN2Conv(hidden_dim, layer=1, activation=F.relu)
        self.gcnii2 = dgl.nn.GCN2Conv(hidden_dim, layer=2, activation=F.relu)

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        h = self.fc(feat)
        res = h
        res = self.gcnii1(g, res, h)
        return self.gcnii2(g, res, h)

class GIN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        def mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
                nn.ReLU()
            )
        self.gin1 = dgl.nn.GINConv(mlp(in_dim, hidden_dim, hidden_dim), learn_eps=True, activation=F.relu)
        self.gin2 = dgl.nn.GINConv(mlp(hidden_dim, hidden_dim, hidden_dim), learn_eps=True, activation=F.relu)

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        h = self.gin1(g, feat)
        return self.gin2(g, h)

class SGC(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(SGC, self).__init__()
        self.conv = dgl.nn.SGConv(in_feats, num_classes, k=1)

    def forward(self, g, in_feat):
        h = self.conv(g, in_feat)
        return h


class APPNP(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(APPNP, self).__init__()
        self.mlp = torch.nn.Linear(in_feats, num_classes)
        self.conv = dgl.nn.APPNPConv(k=1, alpha=0.2)

    def forward(self, g, in_feat):
        in_feat = self.mlp(in_feat)
        h = self.conv(g, in_feat)
        return h


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dgl.nn.SAGEConv(in_feats, hid_feats, aggregator_type='mean', activation=F.relu)
        self.conv2 = dgl.nn.SAGEConv(hid_feats, hid_feats, aggregator_type='mean', activation=F.relu)

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = self.conv2(graph, h)
        return h
