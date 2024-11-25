import opt
from train import *
from DFCGC import DFCGC


if __name__ == '__main__':
    # setup
    # for opt.args.name in ["acm", "dblp", "cite", "amap", "cora"]:
    # for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    # for opt.args.name in ["acm", "dblp", "cite", "amap"]:
    for opt.args.name in ["amap"]:
        for s in range(1):
            setup()
            setup_seed(s)
            # opt.args.lambda_value = c
            # print("lambda:", opt.args.lambda_value)

            # data pre-precessing: X, y, A, A_norm, Ad
            X, y, A = load_graph_data(opt.args.name, show_details=False)
            A_norm = normalize_adj(A, self_loop=True, symmetry=False)
            Ad = diffusion_adj(A, mode="ppr", transport_rate=opt.args.alpha_value)

            features = X
            # to torch tensor
            X = numpy_to_torch(X).to(opt.args.device)
            A_norm = numpy_to_torch(A_norm, sparse=True).to(opt.args.device)
            Ad = numpy_to_torch(Ad).to(opt.args.device)

            # dpc = DPC(X, n=3, dc_method=0, dc_percent=2, rho_method=1, delta_method=1, use_halo=False, plot=None, save_path='plot.jpg')

            # Dual Correlation Reduction Network
            model = DFCGC(n_node=X.shape[0]).to(opt.args.device)
            train(model, X, y, A, A_norm, Ad, features)