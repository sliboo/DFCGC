import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from utils import *

def show_elbow_and_silhouette_score(data_values, file_name_1, file_name_2):
    '''
    1. 计算 Elbow Method 并选择变化率最大的前三个 k 值
    2. 计算 Silhouette Score 并选择分数最高的前三个 k 值
    3. 分别保存为两个 PDF 文件
    4. 输出 Elbow 和 Silhouette Score Method 所选的前三个 k 值
    :param data_values: 聚类数据
    :param file_name_1: 保存 Elbow Method 图的文件名
    :param file_name_2: 保存 Silhouette Score 图的文件名
    :return: Elbow Method 和 Silhouette Score Method 所选择的前三个 k 值
    '''

    if isinstance(data_values, torch.Tensor):
        data_values = data_values.detach().cpu().numpy()

    sse = {}
    sil = []
    kmax = 10

    # Elbow Method:
    plt.figure(figsize=(10, 5))
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, max_iter=1000, random_state=42).fit(data_values)
        sse[k] = kmeans.inertia_

    # 计算相邻 SSE 点的变化率，并选择变化率最大的前三个位置
    sse_diff = np.diff(list(sse.values()))  # SSE 差异
    sse_diff_abs = np.abs(sse_diff)  # 取绝对值来找到变化最大的点
    elbow_top_3_k = np.argsort(sse_diff_abs)[::-1][:3] + 2  # 选择变化率最大的前三个 k 值，并+2调整索引

    sns.lineplot(x=list(sse.keys()), y=list(sse.values()), marker='o')
    plt.title('Elbow Method')
    plt.xlabel("k : Number of clusters")
    plt.ylabel("Sum of Squared Error")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{file_name_1}.pdf", format='pdf')  # 保存 Elbow Method 为 PDF
    plt.show()

    print(f"Elbow Method suggests top 3 k values: {elbow_top_3_k}")

    # # Silhouette Score Method:
    # plt.figure(figsize=(10, 5))
    # for k in range(2, kmax + 1):
    #     kmeans = KMeans(n_clusters=k).fit(data_values)
    #     labels = kmeans.labels_
    #     sil.append(silhouette_score(data_values, labels, metric='euclidean'))
    #
    # sil_top_3_k = np.argsort(sil)[::-1][:3] + 2  # Silhouette 分数最高的前三个 k 值
    #
    # sns.lineplot(x=range(2, kmax + 1), y=sil)
    # plt.title('Silhouette Score Method')
    # plt.xlabel("k : Number of clusters")
    # plt.ylabel("Silhouette Score")
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig(f"{file_name_2}.pdf", format='pdf')  # 保存 Silhouette Score 为 PDF
    # plt.show()
    #
    # print(f"Silhouette Score Method suggests top 3 k values: {sil_top_3_k}")

    return elbow_top_3_k

class DPC(object):
    """
    1. 计算距离
    2. 计算截断距离 dc
    3. 计算局部密度 rho
    4. 计算 delta
    5. 确定 聚类中心
    6. 聚类
    7. 绘图
    """

    def __init__(self, X, dc_percent=1, delta_method=1, plot=False, save_path=None):
        # 使用torch张量计算距离
        points, d_matrix, d_list, min_dis, max_dis, max_id = self.load_points_cacl_distance(X)
        # 计算截断聚类 dc
        dc = self.get_dc(d_matrix, d_list, min_dis, max_dis, max_id, dc_percent)
        # print('dc: ', dc)
        # 计算 rho
        rho = self.get_rho(d_matrix, max_id, dc)
        # 计算 delta
        delta = self.get_delta(d_matrix, max_id, rho, delta_method)
        # 确定聚类中心
        # center, gamma = self.get_center(d_matrix, rho, delta, dc, n, max_id)
        cluster_center = self.select_cluster_centers(rho, delta, opt.args.rho, opt.args.delta)
        # cluster_center = self.find_optimal_clusters_with_rho_delta(X, 11, rho, delta)

        rho_np = rho.detach().cpu().numpy()
        delta_np = delta.detach().cpu().numpy()

        if (plot):
            plt.figure(figsize=(10, 6))

            # 绘制所有数据点（蓝色）
            plt.scatter(rho_np, delta_np, c='blue', alpha=0.5, label='Data points')

            # 单独绘制聚类中心（红色圆圈）
            plt.scatter(rho_np[cluster_center], delta_np[cluster_center], c='red', s=100, edgecolor='black',
                        label='Cluster centers')

            # 可选：使用 plt.annotate 在图中显示聚类中心的索引
            for idx in cluster_center:
                plt.annotate(str(idx), (rho_np[idx], delta_np[idx]), textcoords="offset points", xytext=(0, 5),
                             ha='center',
                             fontsize=8)

            plt.title('Number of Clusters Learning')
            plt.xlabel(r'$\rho$')
            plt.ylabel(r'$\delta$')
            plt.grid(True)
            plt.legend()

            # 保存或显示图像
            if save_path is not None:
                plt.savefig(save_path)
                print(f'Plot saved to {save_path}')
            else:
                plt.show()

            plt.close()  # 关闭图形以释放内存

        self.num_cluster_centers = len(cluster_center)

    def calculate_silhouette_with_rho_delta(self, rho, delta):
        # 将 rho 作为紧密度 a，delta 作为分离度 b
        a = rho
        b = delta
        # 计算每个点的轮廓系数 (b - a) / max(a, b)
        silhouette_values = (b - a) / np.maximum(a, b)
        return silhouette_values

    def find_optimal_clusters_with_rho_delta(self, data, max_k, rho, delta):
        silhouette_scores = []

        # 遍历从2到max_k的不同聚类个数
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k)
            labels = kmeans.fit_predict(data)

            # 使用rho和delta计算的轮廓系数
            silhouette_values = self.calculate_silhouette_with_rho_delta(rho, delta)
            score = silhouette_score(data, labels, metric='euclidean', sample_size=None)

            silhouette_scores.append(score)
            print(f"聚类数: {k}, 轮廓系数: {score:.4f}")

        # 找到轮廓系数最高的聚类个数
        best_k = np.argmax(silhouette_scores) + 2  # 因为range从2开始
        print(f"最佳聚类个数: {best_k}")

        return best_k, silhouette_scores

    def load_points_cacl_distance(self, X):
        # X 是 torch 张量，大小为 [num_points, num_features]
        points = X
        max_id = X.size(0)  # 点的数量

        # 1. 使用 torch 计算欧式距离矩阵 (cdist)
        d_matrix = torch.cdist(X, X, p=2)  # 欧式距离矩阵

        # 2. 取出距离矩阵的上三角部分，不包含对角线
        d_list = d_matrix.triu(diagonal=1).flatten()  # 提取上三角的元素
        d_list = d_list[d_list > 0]  # 过滤掉距离为 0 的元素（自我距离）

        # 3. 获取最小距离和最大距离
        min_dis = d_list.min().item()  # 计算最小值
        max_dis = d_list.max().item()  # 计算最大值

        return points, d_matrix, d_list, min_dis, max_dis, max_id

    def select_cluster_centers(self, rho, delta, rho_threshold=None, delta_threshold=None, percentile=90):
        """
        根据 rho 和 delta 的值自动选择聚类中心。

        参数:
        - rho: 张量或 numpy 数组，表示局部密度。
        - delta: 张量或 numpy 数组，表示与最近高密度点的距离。
        - rho_threshold: float，可选，指定的 rho 阈值，超过该值的点作为候选。
        - delta_threshold: float，可选，指定的 delta 阈值，超过该值的点作为候选。
        - percentile: int，分位数，用于自动确定阈值。

        返回:
        - cluster_centers: int，聚类中心的数量。
        """
        # 确保 rho 和 delta 是 numpy 数组
        if isinstance(rho, torch.Tensor):
            rho = rho.detach().cpu().numpy()  # 将 torch.Tensor 转换为 numpy 数组
        if isinstance(delta, torch.Tensor):
            delta = delta.detach().cpu().numpy()  # 将 torch.Tensor 转换为 numpy 数组

        if rho_threshold is not None:
            rho_threshold = np.percentile(rho, rho_threshold)  # 将百分数转换为分位数
        else:
            rho_threshold = np.percentile(rho, percentile)  # 默认使用提供的分位数

        if delta_threshold is not None:
            delta_threshold = np.percentile(delta, delta_threshold)  # 将百分数转换为分位数
        else:
            delta_threshold = np.percentile(delta, percentile)  # 默认使用提供的分位数

            # 选择满足条件的点作为聚类中心
        cluster_centers = np.where((rho >= rho_threshold) & (delta >= delta_threshold))[0]

        return cluster_centers

    def get_dc(self, d, d_list, min_dis, max_dis, max_id, percent):
        """ 求解截断距离 """
        # print('Get dc')
        lower = percent / 100
        upper = (percent + 1) / 100
        while 1:
            dc = (min_dis + max_dis) / 2
            neighbors_percent = len(d_list[d_list < dc]) / (((max_id - 1) ** 2) / 2)  # 上三角矩阵
            if neighbors_percent >= lower and neighbors_percent <= upper:
                return dc
            elif neighbors_percent > upper:
                max_dis = dc
            elif neighbors_percent < lower:
                min_dis = dc
    def get_rho(self, d_matrix, max_id, dc):
        # 你可能需要将 d_matrix 移动到 CPU 上，如果它是在 GPU 上的张量
        d_matrix_cpu = d_matrix.cpu() if d_matrix.is_cuda else d_matrix
        rho = torch.zeros(max_id)

        # 使用 numpy 计算 rho 的一部分
        for i in range(max_id):
            d = d_matrix_cpu[i]  # 提取第 i 行的距离
            exp_d = np.exp(-((d.detach().cpu() / dc) ** 2))  # 预计算高斯核
            rho[i] = exp_d.sum()  # 计算 rho

        return rho

    def get_delta(self, d, max_id, rho, method):
        """ 获取 delta """
        # print('Get delta')
        delta = torch.zeros(max_id)
        if method == 0:  # 不考虑rho相同且同为最大
            for i in range(max_id):
                rho_i = rho[i]
                j_list = torch.where(rho > rho_i)[0]  # rho 大于 rho_i 的点们
                if len(j_list) == 0:
                    delta[i] = d[i, :].max()
                else:
                    delta[i] = d[i, j_list].min()
        elif method == 1:
            rho_order_index = torch.argsort(rho, descending=True)  # rho 排序索引
            for i in range(1, max_id):
                rho_index = rho_order_index[i]  # 对应 rho 的索引（点的编号）
                j_list = rho_order_index[:i]  # rho 大于 i 的列表
                delta[rho_index] = d[rho_index, j_list].min()
            delta[rho_order_index[0]] = delta.max()
        return delta
