import argparse

parser = argparse.ArgumentParser(description='DCRN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# setting
parser.add_argument('--name', type=str, default="acm")
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--seed', type=int, default=3)
parser.add_argument('--alpha_value', type=float, default=0.2)
parser.add_argument('--lambda_value', type=float, default=10)
parser.add_argument('--gamma_value', type=float, default=1e3)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--n_z', type=int, default=20)
parser.add_argument('--epoch', type=int, default=500)
# parser.add_argument('--rho', type=int, default=40)
# parser.add_argument('--delta', type=int, default=95)
parser.add_argument('--rho', type=int, default=35)
parser.add_argument('--delta', type=int, default=97)
parser.add_argument('--save_dir', type=str, default='tsne')

parser.add_argument('--relr', type=float, default=1e-5, help='Initial learning rate.')

parser.add_argument('--gnnlayers', type=int, default=3, help="Number of gnn layers")
parser.add_argument('--dims', type=int, default=[20], help='Number of units in hidden layer 1.')

# Q net
parser.add_argument('--Q_epochs', type=int, default=30, help='Number of epochs to train Q.')
parser.add_argument('--epsilon', type=float, default=0.5, help='Greedy rate.')
parser.add_argument('--replay_buffer_size', type=float, default=50, help='Replay buffer size')
parser.add_argument('--Q_lr', type=float, default=1e-3, help='Initial learning rate.')


# AE structure parameter from DFCN
parser.add_argument('--ae_n_enc_1', type=int, default=128)
parser.add_argument('--ae_n_enc_2', type=int, default=256)
parser.add_argument('--ae_n_enc_3', type=int, default=512)
parser.add_argument('--ae_n_dec_1', type=int, default=512)
parser.add_argument('--ae_n_dec_2', type=int, default=256)
parser.add_argument('--ae_n_dec_3', type=int, default=128)

# IGAE structure parameter from DFCN
parser.add_argument('--gae_n_enc_1', type=int, default=128)
parser.add_argument('--gae_n_enc_2', type=int, default=256)
parser.add_argument('--gae_n_enc_3', type=int, default=20)
parser.add_argument('--gae_n_dec_1', type=int, default=20)
parser.add_argument('--gae_n_dec_2', type=int, default=256)
parser.add_argument('--gae_n_dec_3', type=int, default=128)

# clustering performance: acc, nmi, ari, f1
parser.add_argument('--acc', type=float, default=0)
parser.add_argument('--nmi', type=float, default=0)
parser.add_argument('--ari', type=float, default=0)
parser.add_argument('--f1', type=float, default=0)

args = parser.parse_args()
