import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataname', type=str, default='cora', help='Name of dataset.')
    parser.add_argument('--epoch1', type=int, default=100, help='Training epochs.')

    parser.add_argument('--lr1', type=float, default=0.001, help='Learning rate of HomoGCL.')
    parser.add_argument('--wd1', type=float, default=0.01, help='Weight decay of HomoGCL.')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--der', type=float, default=0.3, help='Drop edge ratio.') # 0.9
    parser.add_argument('--dfr', type=float, default=0.1, help='Drop feature ratio.')
    parser.add_argument("--hid_dim", type=int, default=512, help='Hidden layer dim.')
    parser.add_argument("--out_dim", type=int, default=256, help='Output layer dim.') # 256
    parser.add_argument("--tau", type=float, default=0.5, help='Temperature')

    # 【Cora】【lr1 0.001】【wd1 0.01】 【n_layers 2】 【der 0.3】 【dfr 0.1】 【hid_dim 512】 【out_dim 256】【tau 0.5】
    # 【Citeseer】【lr1 0.001】【wd1 0.1】 【n_layers 2】 【der 0.7】 【dfr 0.1】 【hid_dim 32】 【out_dim 256】【tau 0.7】
    # 【AMAP】【lr1 0.001】【wd1 0.00001】 【n_layers 3】 【der 0.9】 【dfr 0.1】 【hid_dim 512】 【out_dim 32】【tau 0.3】
    # 【BAT】【lr1 0.001】【wd1 0.001】 【n_layers 9】 【der 0.8】 【dfr 0.8】 【hid_dim 16】 【out_dim 256】【tau 0.5】
    # 【EAT】【lr1 0.001】【wd1 0.1】 【n_layers 8】 【der 0.7】 【dfr 0.5】 【hid_dim 8】 【out_dim 256】【tau 0.8】
    # 【UAT】【lr1 0.001】【wd1 0.0001】 【n_layers 2】 【der 0.8】 【dfr 0.1】 【hid_dim 8】 【out_dim 128】【tau 0.5】	
 
    parser.add_argument('--gpu', type=int, default=0, help='GPU index. -1 for cpu')
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--proj_dim", type=int, default=0, help='Project dim.')
    parser.add_argument('--mean', action="store_true", help='Calculate mean for neighbor pos')
    parser.add_argument("--niter", type=int, default=20, help='Number of iteration for kmeans.')
    parser.add_argument("--sigma", type=float, default=1e-3, help='2sigma^2 in GMM') # 1e-3
    parser.add_argument("--alpha", type=int, default=1, help='Coefficient alpha') # 1
    parser.add_argument("--clustering", action='store_true', default=False, help='Downstream clutering task.')
    parser.add_argument("--repetition_cluster", type=int, default=10, help='Repetition of clustering')
    args = parser.parse_args()
    
    return args
