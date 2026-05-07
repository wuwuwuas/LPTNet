import torch

class KNN:
    def __init__(self, k: int):
        self.k = k

    def __call__(self, y: torch.Tensor, x: torch.Tensor = None):
        """
        x: [B, N1, C]  要查询的点集
        y: [B, N2, C]  被查询的点集，若为 None 则为自查询
        返回：
            dists: [B, N1, k]  每个点到其 k 个最近邻的距离平方
            indices: [B, N1, k]  每个点的 k 个最近邻在 y 中的索引
        """
        if y is None:
            y = x  # 自查询

        B, N1, C = x.shape
        _, N2, _ = y.shape

        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x·y
        x_norm = (x ** 2).sum(dim=2, keepdim=True)  # [B, N1, 1]
        y_norm = (y ** 2).sum(dim=2, keepdim=True)  # [B, N2, 1]
        y_norm = y_norm.permute(0, 2, 1)            # [B, 1, N2]

        inner_prod = torch.bmm(x, y.transpose(1, 2))  # [B, N1, N2]
        dists = x_norm + y_norm - 2 * inner_prod      # [B, N1, N2]

        # 防止数值误差导致负值
        dists = torch.clamp(dists, min=0)

        dists_k, idx = torch.topk(dists, self.k, dim=-1, largest=False, sorted=True)  # [B, N1, k]
        return dists_k, idx



# class KNN:
#     def __init__(self, k: int):
#         self.k = k
#
#     def __call__(self, y: torch.Tensor, x: torch.Tensor = None):
#         """
#         稳定版本的KNN，确保跨平台一致性
#         """
#         if y is None:
#             y = x
#
#         B, N1, C = x.shape
#         _, N2, _ = y.shape
#
#         # 计算距离矩阵
#         x_norm = (x ** 2).sum(dim=2, keepdim=True)
#         y_norm = (y ** 2).sum(dim=2, keepdim=True)
#         y_norm = y_norm.permute(0, 2, 1)
#
#         inner_prod = torch.bmm(x, y.transpose(1, 2))
#         dists = x_norm + y_norm - 2 * inner_prod
#
#         # 防止数值误差
#         dists = torch.clamp(dists, min=1e-12)
#
#         # 添加微小的随机噪声来打破ties，确保稳定排序
#
#         tie_breaker = torch.arange(N2, device=dists.device, dtype=dists.dtype).view(1, 1, -1)
#         tie_breaker = tie_breaker.expand(B, N1, -1) * 1e-10
#         dists = dists + tie_breaker
#
#         dists_k, idx = torch.topk(dists, self.k, dim=-1, largest=False, sorted=True)
#         return dists_k, idx

# B, N1, N2, C = 2, 5, 10, 3
# x = torch.rand(B, N1, C)  # 查询点
# y = torch.rand(B, N2, C)  # 被查询点
#
# knn = KNN(k=4)
# dists, indices = knn(x, y)
#
# print("距离：", dists.shape)     # [2, 5, 4]
# print("索引：", indices.shape)   # [2, 5, 4]
