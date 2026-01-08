import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class NeuralClusterEstimator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(NeuralClusterEstimator, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )
        self.salience_net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        profile = self.regressor(z)
        rho, delta = profile[:, 0], profile[:, 1]
        score = self.salience_net(profile)
        return rho, delta, score


class NCETrainer:
    def __init__(self, input_dim, device):
        self.device = device
        self.model = NeuralClusterEstimator(input_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.cached_rho = None
        self.cached_delta = None


    def _get_dpc_targets(self, Z):
        num_nodes = Z.size(0)
        # 4090 优化：计算距离矩阵
        dist_matrix = torch.cdist(Z, Z)

        # 采样计算 dc 避开 quantile 报错
        if num_nodes > 4000:
            indices = torch.randperm(num_nodes)[:4000]
            sample_dist = dist_matrix[indices][:, indices]
            dc = torch.quantile(sample_dist.view(-1), 0.02)
        else:
            dc = torch.quantile(dist_matrix.view(-1), 0.02)


        rho = torch.sum(torch.exp(-(dist_matrix / dc) ** 2), dim=1)


        rho_sorted, indices = torch.sort(rho, descending=True)
        delta = torch.zeros_like(rho)


        delta.fill_(float('inf'))
        for i in range(1, num_nodes):
            idx = indices[i]
            higher_rho_idx = indices[:i]
            delta[idx] = torch.min(dist_matrix[idx, higher_rho_idx])

        delta[indices[0]] = torch.max(delta)
        return rho, delta

    def update_targets(self, Z):
        print("Updating NCE targets (DPC calculation)...")
        with torch.no_grad():
            # 确保这里调用的是 self._get_dpc_targets
            rho, delta = self._get_dpc_targets(Z.detach())
            self.cached_rho = rho
            self.cached_delta = delta

    def train_step_fast(self, Z):
        if self.cached_rho is None:
            self.update_targets(Z)

        self.model.train()
        pred_rho, pred_delta, scores = self.model(Z.detach())

        loss = F.mse_loss(pred_rho, self.cached_rho) + \
               F.mse_loss(pred_delta, self.cached_delta)
        loss += 0.01 * torch.mean(scores)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict_k(self, Z, threshold=0.5):
        self.model.eval()
        with torch.no_grad():
            _, _, scores = self.model(Z)
            k = torch.sum(scores > threshold).item()
        return max(int(k), 2)