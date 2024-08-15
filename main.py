import torch as th
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
from aug import random_aug
from dataset import load,load1,load2
from model import HomoGCL, EnDecoder
from utils import cal_homo_ratio
from args import parse_args
from sklearn.cluster import KMeans
from evaluate_embedding import cluster_acc
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score,f1_score,recall_score,precision_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings('ignore')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

def log_results(epoch, acc, nmi, ari, f1, file_path='training_log.txt'):
    log_message = f'Iter {epoch}: Acc {acc:.4f}, nmi {nmi:.4f}, ari {ari:.4f}, f1 {f1:.4f}'
    with open(file_path, 'a') as f:
        f.write(log_message + '\n')

args = parse_args()
set_seed(args.seed)

#if args.gpu != -1 and th.cuda.is_available():
#    args.device = 'cuda:{}'.format(args.gpu)
#else:
args.device = 'cpu'

if __name__ == '__main__':
    args.alpha = 1.0
    print(args)
    if 'texas'==args.dataname or 'cornell'==args.dataname or 'washington'==args.dataname or 'wisconsin'==args.dataname:
        graph, feat, labels, num_class, adjs_labels, graph_num, shared_feature_label = load2(args.dataname)
    else:
        graph, feat, labels, num_class, adjs_labels, graph_num, shared_feature_label = load1(args.dataname)
    
    for v in range(graph_num):
        r = cal_homo_ratio(adjs_labels[v].cpu().numpy(), labels.cpu().numpy(), self_loop=True)
        print('adjs_labels[v] dim is: ', adjs_labels[v].size())
        print(args.dataname +' Homophily Ratio: ',r)
    
    in_dim = feat.shape[1]
    N = graph.number_of_nodes()
    # print("in_dim: ",in_dim)
    # print("N: ", N)
    model = HomoGCL(in_dim, args.hid_dim, args.out_dim, args.n_layers, N, num_proj_hidden=args.proj_dim, tau=args.tau)
    model = model.to(args.device)
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)

    endecoder = EnDecoder(in_dim, args.hid_dim, args.out_dim, num_class)
    endecoder = endecoder.to(args.device)
    optimizer_endecoder = th.optim.Adam(endecoder.parameters(), lr=args.lr1, weight_decay=args.wd1)

    graph_cuda = graph.to(args.device)
    graph_cuda = graph_cuda.remove_self_loop().add_self_loop()
    feat_cuda = feat.to(args.device)

    loss_all = []
    for epoch in range(1, args.epoch1+1):
        model.train()
        endecoder.train()
        optimizer.zero_grad()
        optimizer_endecoder.zero_grad()

        a_pred, x_pred, z_norm, S = endecoder(feat)
        loss_re = F.binary_cross_entropy(x_pred, shared_feature_label)
        
        graph1, feat1 = random_aug(graph, feat, args.dfr, args.der) 
        graph2, feat2 = random_aug(graph, feat, args.dfr, args.der)
        graph1 = graph1.add_self_loop()
        graph2 = graph2.add_self_loop()
        graph1 = graph1.to(args.device)
        graph2 = graph2.to(args.device)
        feat1 = feat1.to(args.device)
        feat2 = feat2.to(args.device)
        z1, z2, z = model(graph1, feat1, graph2, feat2, graph_cuda, feat_cuda)
        adj1 = th.zeros(N, N, dtype=th.int).to(args.device)
        adj1[graph1.remove_self_loop().edges()] = 1
        adj2 = th.zeros(N, N, dtype=th.int).to(args.device)
        adj2[graph2.remove_self_loop().edges()] = 1

        compare_loss, h1, h2 = model.loss(z1, adj1, z2, adj2, S, args.mean)
        kl_loss = model.kl_loss(h1, h2)
        
        loss = loss_re + compare_loss + kl_loss
        if epoch % 10 == 0:
            loss_all.append(loss.detach().numpy())

        loss.backward()
        optimizer.step()
        optimizer_endecoder.step()

        print('Epoch={:03d}, loss={:.10f}'.format(epoch, loss.item()))

        if epoch % 10 == 0:
            model.eval()
            with th.no_grad():
                embeds = model.get_embedding(graph_cuda, feat_cuda)
                kmeans = KMeans(n_clusters=num_class, n_init=100)

                y_pred = kmeans.fit_predict(embeds)
                label = labels.to(args.device).cpu().numpy()
                print('y_pred',y_pred)
                print('label',label)

                acc, f1 = cluster_acc(label, y_pred)
                nmi = nmi_score(label, y_pred, average_method='arithmetic')
                ari = ari_score(label, y_pred)
                
                print('===== Clustering performance: =====')
                print('Iter {}'.format(epoch), ':Acc {:.4f}'.format(acc),
                    ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari), ', f1 {:.4f}'.format(f1))

                if epoch == args.epoch1:
                    log_results(epoch, acc, nmi, ari, f1)

                    tsne = TSNE(n_components=2, random_state=42)
                    X_tsne = tsne.fit_transform(embeds)
                    for cluster_label in range(num_class):
                        plt.scatter(X_tsne[y_pred == cluster_label, 0], X_tsne[y_pred == cluster_label, 1], label=f'Cluster {cluster_label}')
                    #plt.legend()
                    plt.savefig('image/'+str(args.dataname)+str(args.seed)+'.png')

    loss_array = np.array(loss_all)
    np.save('loss_data.npy', loss_array)
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(0, args.epoch1, 10), loss_all, label='Total Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('loss_convergence.png', dpi=300)