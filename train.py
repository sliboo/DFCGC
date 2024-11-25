import tqdm

import opt
from utils import *
from torch.optim import Adam
import scipy.sparse as sp
import os
import matplotlib
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os.path as osp
import os
from datetime import datetime
import time

matplotlib.use('Agg')
torch.cuda.empty_cache()  # 手动清理显存
from dpc import DPC, show_elbow_and_silhouette_score
from dbscan import MyDBSCAN, calculate_optimal_eps
from dpca import DensityPeakCluster

def plot_embedding(data, label, title):

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    unique_labels = np.unique(label)
    num_labels = len(unique_labels)
    color_map = ['r','y','g','b','m','c','k','orange','pink','gold','cyan']

    fig = plt.figure()
    for i in range(data.shape[0]):
        c = color_map[label[i]]
        plt.plot(data[i, 0], data[i, 1], marker='o', markersize=1, c=c)
    plt.xticks([])
    plt.yticks([])
    plt.title(opt.args.name+'-DFCGC')
    dirname = osp.join(opt.args.save_dir, opt.args.name)
    if not osp.exists(dirname):
        os.mkdir(dirname)
    save_name = osp.join(dirname, opt.args.name + str(title) + '.pdf')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()

def train(model, X, y, A, A_norm, Ad, features):
    """
    train our model
    Args:
        model: Dual Correlation Reduction Network
        X: input feature matrix
        y: input label
        A: input origin adj
        A_norm: normalized adj
        Ad: graph diffusion
    Returns: acc, nmi, ari, f1
    """
    print("Training…")
    file_name = "result.csv"
    file = open(file_name, "a+")
    print(opt.args.name, file=file)
    file.close()
    tsne = TSNE(n_components=2, random_state=0)

    sim, centers, Z = model_init(model, X, y, A_norm)

    # initialize cluster centers
    model.cluster_centers.data = torch.tensor(centers).to(opt.args.device)

    # edge-masked adjacency matrix (Am): remove edges based on feature-similarity
    Am = remove_edge(A, sim, remove_rate=0.1)

    best_nmi = 0
    best_ari = 0
    best_cluster = 0

    # eps_value = 12 / 10.0
    # _, c = MyDBSCAN(Z.detach().cpu().numpy(), eps=eps_value, MinPts=2)
    # print(c)
    # opt.args.n_clusters = c

    # dpca = DensityPeakCluster()
    # labels = dpca.fit(Z.detach().cpu().numpy())
    # print(f'聚类个数: {labels}')
    # opt.args.n_clusters = labels

    best_nmi, best_ari, _ = clustering2(Z.detach(), y, opt.args.n_clusters)
    # print(opt.args.n_clusters, best_nmi, best_ari)

    dcp = DPC(Z, dc_percent=2, delta_method=1, plot=False, save_path="z_0")
    opt.args.n_clusters = dcp.num_cluster_centers
    print(dcp.num_cluster_centers)

    optimizer = Adam(model.parameters(), lr=opt.args.lr)

    for epoch in tqdm.tqdm(range(opt.args.epoch)):
    # for epoch in range(opt.args.epoch):
        model.train()
        X_tilde1, X_tilde2 = gaussian_noised_feature(X)

        # input & output
        X_hat, Z_hat, A_hat, si, Z_ae_all, Z_gae_all, Q, Z, AZ_all, Z_all = model(X_tilde1, Ad, X_tilde2, Am)

        # if (epoch+1) % 50 == 0:
        #     data_cpu = Z.cpu().detach().numpy()
        #     data_tsne = tsne.fit_transform(data_cpu)
        #     labels = Q[0].detach().cpu().numpy().argmax(1)
        #
        #     # 可视化降维后的数据
        #     plot_embedding(data_tsne, labels, epoch)
        #     print("{} is ok.".format(epoch + 1))

        if (epoch > 98 and (epoch + 1) % 50 == 0):
            z_save_path = f'z_epoch_{epoch}.pdf'

            dcp = DPC(Z, dc_percent=2, delta_method=1, plot=False, save_path=z_save_path)
            nmi, ari, _ = clustering2(Z.detach(), y, dcp.num_cluster_centers)
            print(dcp.num_cluster_centers, nmi)

            if epoch > 199:
                if nmi > best_nmi:
                    opt.args.n_clusters = dcp.num_cluster_centers
            else:
                opt.args.n_clusters = dcp.num_cluster_centers

        nmi, ari, _ = clustering2(Z.detach(), y, opt.args.n_clusters)

        # calculate loss: L_{DICR}, L_{REC} and L_{KL}
        L_DICR = dicr_loss(Z_ae_all, Z_gae_all, AZ_all, Z_all)
        L_REC = reconstruction_loss(X, A_norm, X_hat, Z_hat, A_hat)
        L_KL = distribution_loss(Q, target_distribution(Q[0].data))
        loss = L_DICR + L_REC + opt.args.lambda_value * L_KL

        if nmi > best_nmi:
            best_nmi = nmi
            best_ari = ari
            best_cluster = opt.args.n_clusters

        # optimization
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()


    print(best_cluster, best_nmi, best_ari)
    file = open(file_name, "a+")
    print(opt.args.lambda_value, file=file)
    print(best_cluster, file=file)
    print(best_nmi, file=file)
    print(best_ari, file=file)
    file.close()
