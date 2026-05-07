import torch


class Graph:
    def __init__(self, edges, edge_feats, k_neighbors, size):
        """
        Directed nearest neighbor graph constructed on a point cloud.

        Parameters
        ----------
        edges : torch.Tensor
            Contains list with nearest neighbor indices.
        edge_feats : torch.Tensor
            Contains edge features: relative point coordinates.
        k_neighbors : int
            Number of nearest neighbors.
        size : tuple(int, int)
            Number of points.

        """

        self.edges = edges
        self.size = tuple(size)
        self.edge_feats = edge_feats
        self.k_neighbors = k_neighbors

    @staticmethod
    def construct_graph(pcloud, nb_neighbors):
        """
        Construct a directed nearest neighbor graph on the input point cloud.

        Parameters
        ----------
        pcloud : torch.Tensor
            Input point cloud. Size B x N x 3.
        nb_neighbors : int
            Number of nearest neighbors per point.

        Returns
        -------
        graph : flot.models.graph.Graph
            Graph build on input point cloud containing the list of nearest
            neighbors (NN) for each point and all edge features (relative
            coordinates with NN).

        """

        # Size
        nb_points = pcloud.shape[1]
        size_batch = pcloud.shape[0]

        # Distance between points
        distance_matrix = torch.sum(pcloud ** 2, -1, keepdim=True)
        distance_matrix = distance_matrix + distance_matrix.transpose(1, 2)
        distance_matrix = distance_matrix - 2 * torch.bmm(
            pcloud, pcloud.transpose(1, 2)
        )

        # Find nearest neighbors
        neighbors = torch.argsort(distance_matrix, -1)[..., :nb_neighbors]
        effective_nb_neighbors = neighbors.shape[-1]
        neighbors = neighbors.reshape(size_batch, -1)

        # Edge origin
        idx = torch.arange(nb_points, device=distance_matrix.device).long()
        idx = torch.repeat_interleave(idx, effective_nb_neighbors)

        # Edge features
        edge_feats = []
        for ind_batch in range(size_batch):
            edge_feats.append(
                pcloud[ind_batch, neighbors[ind_batch]] - pcloud[ind_batch, idx]
            )
        edge_feats = torch.cat(edge_feats, 0)

        # Handle batch dimension to get indices of nearest neighbors
        for ind_batch in range(1, size_batch):
            neighbors[ind_batch] = neighbors[ind_batch] + ind_batch * nb_points
        neighbors = neighbors.view(-1)

        # Create graph
        graph = Graph(
            neighbors,
            edge_feats,
            effective_nb_neighbors,
            [size_batch * nb_points, size_batch * nb_points],
        )

        return graph


class ParGraph:
    def __init__(self, edges, edge_feats, k_neighbors, size):
        """
        Directed nearest neighbor graph constructed on a point cloud.

        Parameters
        ----------
        edges : torch.Tensor
            Contains list with nearest neighbor indices.
        edge_feats : torch.Tensor
            Contains edge features: relative point coordinates.
        k_neighbors : int
            Number of nearest neighbors.
        size : tuple(int, int)
            Number of points.

        """

        self.edges = edges
        self.edge_feats = edge_feats
        self.k_neighbors = k_neighbors
        self.size = tuple(size)
    @staticmethod
    def construct_graph(pcloud, nb_neighbors):
        """
        Construct a directed nearest neighbor graph on the input point cloud.

        Parameters
        ----------
        pcloud : torch.Tensor
            Input point cloud. Size B x N x 3.
        nb_neighbors : int
            Number of nearest neighbors per point.

        Returns
        -------
        graph : flot.models.graph.Graph
            Graph build on input point cloud containing the list of nearest 
            neighbors (NN) for each point and all edge features (relative 
            coordinates with NN).
            
        """

        # Size
        nb_points = pcloud.shape[1]
        size_batch = pcloud.shape[0]

        # Distance between points
        distance_matrix = torch.sum(pcloud ** 2, -1, keepdim=True) # 求该点距离原点的距离， B N 1
        distance_matrix = distance_matrix + distance_matrix.transpose(1, 2) # .transpose矩阵的转置 B 1 N；B N 1 + B 1 N = B N N
        distance_matrix = distance_matrix - 2 * torch.bmm(  # torch.bmm计算矩阵乘积的函数，要求矩阵为三维 得到B N N 即N个点距离其他点的距离，到自身为0
            pcloud, pcloud.transpose(1, 2))
        # distance_matrix形状为[b n n]就是每个点到其余所有点的距离
        # Find nearest neighbors
        neighbors = torch.argsort(distance_matrix, -1)[..., :nb_neighbors] # torch.argsort()返回只是排序后的值所对应原输入input的下标
        # neighbors存储每个点最近邻点的索引序号，由近到远排序，包括点自身
        effective_nb_neighbors = neighbors.shape[-1] #有效的最近邻点
        neighbors = neighbors.reshape(size_batch, -1)  #reshape为 B N*neighbors_num

        # Edge origin
        idx = torch.arange(nb_points, device=distance_matrix.device).long() # 生成1到nb_points的索引序号
        idx = torch.repeat_interleave(idx, effective_nb_neighbors)# 把索引复制几次
        # idx为[0,0,...,num个0，1,1,...,num个1，...]
        # Edge features
        edge_feats = []
        for ind_batch in range(size_batch):  # 构建点的连边
            edge_feats.append(
                pcloud[ind_batch, neighbors[ind_batch]] - pcloud[ind_batch, idx]
            )
        edge_feats = torch.cat(edge_feats, 0) # 边特征，形状为 B N*neighbors_num 3
        # 存储每个点的最近邻点相对于该点的坐标，也就是以该点为中心创建一个球坐标系，形状为 B*N*num 3按顺序存储的
        feat_r = torch.norm(edge_feats, p=2, dim=-1, keepdim=True) # B,N*nb,1  # torch.norm按维度dim求p范数得到边的长度  论文中的r
        # 存储每条边的长度，形状为 B N*neighbors_num 1
        feat_theta = torch.acos(edge_feats[:,-1].unsqueeze(-1) / (feat_r+0.0001))  # torch.acos反余弦函数   在球坐标系下构建图特征 论文中的角度西塔
        # 存储球坐标系下的角度特征，形状为 B N*neighbors_num 1
        feat_phi = torch.atan(edge_feats[:,1].unsqueeze(-1) / (edge_feats[:,0].unsqueeze(-1)+0.0001))  # 在球坐标系下构建图特征 论文中的角度西塔
        edge_feats_en = torch.cat((edge_feats, feat_r, feat_theta, feat_phi), dim=-1)  # 包含坐标、边长等特征， 形状为 B N*nb 3+1+1+1

        # Handle batch dimension to get indices of nearest neighbors
        for ind_batch in range(1, size_batch):
            neighbors[ind_batch] = neighbors[ind_batch] + ind_batch * nb_points
        neighbors = neighbors.view(-1)

        # Create graph
        graph = ParGraph(
            neighbors, # neighbors存储每个点最近邻点的索引序号，由近到远排序，包括点自身，形状为B*N*num,
            edge_feats_en, # edge_feats_en存储每个点构建的图的信息，形状为B*N*num 6(也就是特征数量，包括图中各点的相对坐标，边的长度(包括该点自身到自身)，球坐标系下的角度信息等) 形状为 B N*nb 3+1+1+1
            effective_nb_neighbors, # 构建的图时，选取的最近邻点的数量
            [size_batch * nb_points, size_batch * nb_points], # B*N B*N
        )

        return graph
