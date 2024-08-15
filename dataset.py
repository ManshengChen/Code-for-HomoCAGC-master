import numpy as np
import torch as th
import scipy.sparse as sp
import pandas as pd

from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset
from dgl import DGLGraph

def load(name):
    if name == 'cora':
        dataset = CoraGraphDataset()
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif name == 'pubmed':
        dataset = PubmedGraphDataset()
    elif name == 'photo':
        dataset = AmazonCoBuyPhotoDataset()
    elif name == 'comp':
        dataset = AmazonCoBuyComputerDataset()
    else:
        raise NotImplementedError
    graph_num = len(dataset)
    print('graph_num',graph_num)
    graph = dataset[0]
    print('graph',graph)
    
    num_class = dataset.num_classes
    feat = graph.ndata.pop('feat')
    labels = graph.ndata.pop('label')
    
    src, dst = graph.edges()
    adj = sp.coo_matrix((np.ones(len(src)), (src.numpy(), dst.numpy())), shape=(graph.number_of_nodes(), graph.number_of_nodes()))
    adj = adj + sp.eye(adj.shape[0])
    adj_tensor = th.FloatTensor(adj.todense())
    adjs_labels = [adj_tensor]

    feature1_label = th.matmul(feat, feat.t())
    feature_labels = th.stack([feat, feature1_label[:feat.size(0), :feat.size(1)]], dim=0)
    shared_feature_label = th.mean(feature_labels, dim=0)
    return graph, feat, labels, num_class, adjs_labels, graph_num, shared_feature_label

def load1(name):
    adj = np.load('data/'+name+'/'+name+'_adj.npy')
    feat = np.load('data/'+name+'/'+name+'_feat.npy')
    labels = np.load('data/'+name+'/'+name+'_label.npy')
    
    graph_num = 1
    node_num = len(adj)
    num_class = len(np.unique(labels))

    print('node_num',node_num)
    print('num_class',num_class)

    edges = np.nonzero(adj)
    print('edges',edges)
    
    graph = DGLGraph()
    graph.add_nodes(node_num)
    graph.add_edges(edges[0], edges[1])
    graph.ndata['feat'] = th.tensor(feat, dtype=th.float32)
    graph.ndata['label'] = th.tensor(labels, dtype=th.int64)
    
    src, dst = graph.edges()
    adj = sp.coo_matrix((np.ones(len(src)), (src.numpy(), dst.numpy())), shape=(graph.number_of_nodes(), graph.number_of_nodes()))
    adj = adj + sp.eye(adj.shape[0])
    adj_tensor = th.FloatTensor(adj.todense())
    adjs_labels = [adj_tensor]
    
    feat = graph.ndata.pop('feat')
    labels = graph.ndata.pop('label')
    feature1_label = th.matmul(feat, feat.t())
    feature_diag = th.diag(feature1_label)
    feature_labels = th.stack([feat, feature_diag.unsqueeze(1).expand_as(feat)], dim=0)
    shared_feature_label = th.sigmoid(th.mean(feature_labels, dim=0))
    return graph, feat, labels, num_class, adjs_labels, graph_num, shared_feature_label

def load2(name):
    node_list = []
    feature_list = []
    labels = []
    df = pd.read_csv('data/'+name+'/raw/out1_node_feature_label.txt', sep='\t', header=None, names=['node_id', 'feature', 'label'])
    for index, row in df.iterrows():
        if index == 0 :
            continue
        labels.append(int(row['label']))
        node_list.append(int(row['node_id']))
        feature_list.append([float(x) for x in row['feature'].split(',')])
    
    edges_df = pd.read_csv('data/'+name+'/raw/out1_graph_edges.txt', sep='\t', header=None, skiprows=1, names=['node_id1', 'node_id2'], dtype=int)
    node_id1 = edges_df['node_id1'].values
    node_id2 = edges_df['node_id2'].values
    edges = (node_id1, node_id2)
    print('edges',edges)

    graph_num = 1
    node_num = max(node_list)
    num_class = len(set(labels))
    
    print('node_num',node_num)
    print('num_class',num_class)

    graph = DGLGraph()
    graph.add_nodes(node_num)
    graph.add_edges(edges[0], edges[1])
    graph.ndata['feat'] = th.tensor(feature_list, dtype=th.float32)
    graph.ndata['label'] = th.tensor(labels, dtype=th.int64)
    
    src, dst = graph.edges()
    adj = sp.coo_matrix((np.ones(len(src)), (src.numpy(), dst.numpy())), shape=(graph.number_of_nodes(), graph.number_of_nodes()))
    adj = adj + sp.eye(adj.shape[0])
    adj_tensor = th.FloatTensor(adj.todense())
    adjs_labels = [adj_tensor]
    
    feat = graph.ndata.pop('feat')
    labels = graph.ndata.pop('label')
    feature1_label = th.matmul(feat, feat.t())
    feature_diag = th.diag(feature1_label)
    feature_labels = th.stack([feat, feature_diag.unsqueeze(1).expand_as(feat)], dim=0)
    shared_feature_label = th.sigmoid(th.mean(feature_labels, dim=0))
    return graph, feat, labels, num_class, adjs_labels, graph_num, shared_feature_label
