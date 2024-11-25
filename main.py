import opt
from train import *
from DFCGC import DFCGC


if __name__ == '__main__':
    for opt.args.name in ["amap"]:
        for s in range(1):
            setup()
            setup_seed(s)
            X, y, A = load_graph_data(opt.args.name, show_details=False)
            A_norm = normalize_adj(A, self_loop=True, symmetry=False)
            Ad = diffusion_adj(A, mode="ppr", transport_rate=opt.args.alpha_value)

            features = X
            X = numpy_to_torch(X).to(opt.args.device)
            A_norm = numpy_to_torch(A_norm, sparse=True).to(opt.args.device)
            Ad = numpy_to_torch(Ad).to(opt.args.device)

            model = DFCGC(n_node=X.shape[0]).to(opt.args.device)
            train(model, X, y, A, A_norm, Ad, features)
