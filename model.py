import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import faiss
from torch.nn.parameter import Parameter
from dgl.nn import GraphConv
from utils import sim
def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

class LatentMappingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6):
        super(LatentMappingLayer, self).__init__()
        self.num_layers = num_layers
        self.enc = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim)
        ])
        for i in range(1, num_layers):
            if i == num_layers - 1:
                self.enc.append(nn.Linear(hidden_dim, output_dim))
            else:
                self.enc.append(nn.Linear(hidden_dim, hidden_dim))
    
    def forward(self, x, dropout=0.1):
        z = self.encode(x, dropout)
        return z
    
    def encode(self, x, dropout=0.1):
        h = x
        for i, layer in enumerate(self.enc):
            if i == self.num_layers - 1:
                if dropout:
                    h = th.dropout(h, dropout, train=self.training)
                h = layer(h)
            else:
                if dropout: 
                    h = th.dropout(h, dropout, train=self.training)
                h = layer(h)
                h = F.tanh(h)
        return h

class EnDecoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim, latent_dim, class_num):
        super(EnDecoder, self).__init__()

        self.enc = LatentMappingLayer(feat_dim, hidden_dim, latent_dim, num_layers=2)
        self.dec_f = LatentMappingLayer(latent_dim, hidden_dim, feat_dim, num_layers=2)
        self.class_num = class_num
    def softDistribution(self, emb, nclusters, niter, sigma):
        kmeans = faiss.Kmeans(emb.shape[1], nclusters, niter=niter) 
        kmeans.train(emb.cpu().detach().numpy())
        centroids = th.FloatTensor(kmeans.centroids).to(emb.device)
        logits = []
        for c in centroids:
            logits.append((-th.square(emb - c).sum(1)/sigma).view(-1, 1))
        logits = th.cat(logits, axis=1)
        probs = F.softmax(logits, dim=1)
        return probs
    
    def forward(self, x, dropout=0.1):
        z = self.enc(x, dropout)
        z_norm = F.normalize(z, p=2, dim=1)
        x_pred = th.sigmoid(self.dec_f(z_norm, dropout))
        a_pred = th.sigmoid(th.mm(z, z.t()))

        R = self.softDistribution(z_norm, self.class_num, 20, sigma=0.001)
        S = sim(R, R)

        return a_pred, x_pred, z_norm, S

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super().__init__()

        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        self.convs.append(GraphConv(in_dim, hid_dim, norm='both'))

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GraphConv(hid_dim, hid_dim, norm='both'))
            self.convs.append(GraphConv(hid_dim, out_dim, norm='both'))

    def forward(self, graph, x, edge_weight=None):

        for i in range(self.n_layers - 1):
            x = F.relu(self.convs[i](graph, x, edge_weight=edge_weight))
        x = self.convs[-1](graph, x)

        return x

class HomoGCL(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, num_nodes, num_proj_hidden, tau: float=0.5):
        super(HomoGCL, self).__init__()
        self.out_dim = out_dim
        self.encoder = GCN(in_dim, hid_dim, out_dim, n_layers)
        self.tau: float = tau
        self.fc1 = th.nn.Linear(out_dim, num_proj_hidden)
        self.fc2 = th.nn.Linear(num_proj_hidden, out_dim)
        self.num_nodes = num_nodes
        self.num_proj_hidden = num_proj_hidden
        
        self.neighboraggr = GraphConv(num_nodes, num_nodes, norm='both', weight=False, bias=False)

        self.block = nn.Sequential(
            nn.Linear(out_dim, int(out_dim/2)),
            nn.ReLU(),
            nn.Linear(int(out_dim/2), out_dim),
            nn.ReLU(),
        )
        self.linear_shortcut = nn.Linear(out_dim, out_dim)
        self.alpha = 1.0
        self.cluster_layer = Parameter(th.Tensor(out_dim, out_dim))
        th.nn.init.xavier_uniform_(self.cluster_layer.data)
    
    def posaug(self, graph, x, edge_weight):
        return self.neighboraggr(graph, x, edge_weight=edge_weight)
    
    def forward(self, graph1, feat1, graph2, feat2, graph, feat):
        z1 = self.encoder(graph1, feat1)
        z2 = self.encoder(graph2, feat2)
        z = self.encoder(graph, feat)
        return z1, z2, z
    
    def soft_distribution(self,x):
        q = 1.0 / (1.0 + th.sum(th.pow(x.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / th.sum(q, 1)).t()
        p = target_distribution(q)
        return q,p
    
    def kl_loss(self, h1, h2):
        z1z2 = th.cat((h1, h2), dim=1)
        unfolded = z1z2.view(2, len(z1z2), self.out_dim).transpose(0, 1)
        z1z2 = unfolded.mean(dim=1)
        print('h1.shape',h1.shape)
        z = self.block(z1z2) + self.linear_shortcut(z1z2)
        Qz,P = self.soft_distribution(z)
        Qu,_ = self.soft_distribution(h1)
        Qv,_ = self.soft_distribution(h2)
        Qz_kl_loss = F.kl_div(Qz.log(), P)
        Qu_kl_loss = F.kl_div(Qu.log(), P)
        Qv_kl_loss = F.kl_div(Qv.log(), P)
        return Qz_kl_loss + Qu_kl_loss + Qv_kl_loss

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    
    def semi_loss(self, z1, adj1, z2, adj2, confmatrix, mean):
        f = lambda x: th.exp(x / self.tau)
        refl_sim = f(sim(z1, z1))          
        between_sim = f(sim(z1, z2))   
        if mean:
            pos = between_sim.diag() + (refl_sim * adj1 * confmatrix).sum(1) / (adj1.sum(1)+0.01) 
        else:
            pos = between_sim.diag() + (refl_sim * adj1 * confmatrix).sum(1)
        neg = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag() - (refl_sim * adj1).sum(1) - (between_sim * adj2).sum(1)
        loss = -th.log(pos / (pos + neg))
        return loss
    
    def loss(self, z1, graph1, z2, graph2, confmatrix, mean):
        if self.num_proj_hidden > 0:
            h1 = self.projection(z1)
            h2 = self.projection(z2)
        else:
            h1 = z1
            h2 = z2
        l1 = self.semi_loss(h1, graph1, h2, graph2, confmatrix, mean)
        l2 = self.semi_loss(h2, graph2, h1, graph1, confmatrix, mean)
        ret = (l1 + l2) * 0.5
        ret = ret.mean()
        return ret, h1, h2

    def get_embedding(self, graph, feat):
        with th.no_grad():
            out = self.encoder(graph, feat)
            return out.detach()

class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret

