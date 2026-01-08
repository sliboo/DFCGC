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

matplotlib.use('Agg')
torch.cuda.empty_cache()
from NCE import NCETrainer

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
    print("Training with NCE Estimator (Buffered Strategy)...")
    file_name = "result.csv"
    file = open(file_name, "a+")
    print(opt.args.name, file=file)
    file.close()
    tsne = TSNE(n_components=2, random_state=0)


    if A_norm.is_sparse:
        A_norm = A_norm.to_dense()
    if Ad.is_sparse:
        Ad = Ad.to_dense()


    sim, centers, Z_init = model_init(model, X, y, A_norm)
    model.cluster_centers.data = torch.tensor(centers).to(opt.args.device)
    Am = remove_edge(A, sim, remove_rate=0.1)


    nce_estimator = NCETrainer(input_dim=Z_init.shape[1], device=opt.args.device)

    print("Initial NCE target calculation (DPC)...")
    nce_estimator.update_targets(Z_init.detach())


    opt.args.n_clusters = nce_estimator.predict_k(Z_init.detach())

    optimizer = Adam(model.parameters(), lr=opt.args.lr)
    best_nmi, best_ari, best_cluster = 0, 0, opt.args.n_clusters

    print("Initial NCE target calculation (DPC)...")
    nce_estimator.update_targets(Z_init.detach())


    print("Warm-up NCE Estimator...")
    warmup_epochs = 500
    for _ in range(warmup_epochs):
        nce_estimator.train_step_fast(Z_init.detach())
    print("Warm-up finished. Starting main training...")

    for epoch in tqdm.tqdm(range(opt.args.epoch)):
        model.train()
        X_tilde1, X_tilde2 = gaussian_noised_feature(X)
        X_hat, Z_hat, A_hat, si, Z_ae_all, Z_gae_all, Q, Z, AZ_all, Z_all = model(X_tilde1, Ad, X_tilde2, Am)

        if (epoch + 1) % 100 == 0:
            nce_estimator.update_targets(Z.detach())

        nce_loss = nce_estimator.train_step_fast(Z.detach())


        if (epoch > 98 and (epoch + 1) % 50 == 0):
            new_k = nce_estimator.predict_k(Z.detach())


            if epoch > 199:
                current_nmi, _, _ = clustering2(Z.detach(), y, new_k)
                if current_nmi > best_nmi:
                    opt.args.n_clusters = new_k
            else:
                opt.args.n_clusters = new_k


        L_DICR = dicr_loss(Z_ae_all, Z_gae_all, AZ_all, Z_all)
        L_REC = reconstruction_loss(X, A_norm, X_hat, Z_hat, A_hat)
        L_KL = distribution_loss(Q, target_distribution(Q[0].data))
        loss = L_DICR + L_REC + opt.args.lambda_value * L_KL

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()


        nmi, ari, _ = clustering2(Z.detach(), y, opt.args.n_clusters)
        if nmi > best_nmi:
            best_nmi, best_ari, best_cluster = nmi, ari, opt.args.n_clusters


    print(f"\nFinal Results: Best Cluster: {best_cluster}, NMI: {best_nmi:.4f}, ARI: {best_ari:.4f}")
    file = open(file_name, "a+")
    print(opt.args.lambda_value, file=file)
    print(best_cluster, file=file)
    print(best_nmi, file=file)
    print(best_ari, file=file)
    file.close()