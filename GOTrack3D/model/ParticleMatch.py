import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from .extractor import FlotEncoder, FlotGraph
from .corr2 import CorrBlock2
from model.update import UpdateBlock
from model.scale import KnnDistance
import model.ot as ot
from tools import track_ot as track_ot
from model.model_dgcnn import GeoDGCNN_flow2
from model.model_track import get_recon_flow, get_recon_flow_knn, similarity_verify, removeoutlier, griddata_flow

from .KNN_torch import KNN

class Tracking(nn.Module):
    def __init__(self, args):
        super(Tracking, self).__init__()
        self.hidden_dim = 64
        self.context_dim = 64
        self.feature_extractor = GeoDGCNN_flow2(k=32, emb_dims=1024, dropout=0.5)  # k=32
        self.context_extractor = FlotEncoder()

        self.corr_block = CorrBlock2(num_levels=args.corr_levels, base_scale=args.base_scales,
                                     resolution=3, truncate_k=args.truncate_k)
        self.update_block = UpdateBlock(hidden_dim=self.hidden_dim)

        self.scale_offset = nn.Parameter(torch.ones(1) / 2)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.epsilon = nn.Parameter(torch.zeros(1))

        self.nb_iter = args.nb_iter
        self.mode = args.tracking_mode
        self.candidates = args.candidates
        
        self.sim_neighbors = args.neighbor_similarity
        self.threshold_similarity = args.threshold_similarity
        
        self.outlier_neighbors = args.neighbor_outlier
        self.threshold_outlier = args.threshold_outlier
        
        # self.dve = args.dve
    def matching(self, p, flow_pred, fmap1, fmap2, iters_match):
        [xy1, xy2] = p
        xy11 = xy1 + flow_pred 
        b, nb_points1, c = xy1.shape
        dis_range = [1, 1]
        knn_sim = KNN(k=self.sim_neighbors+1)
        knn_candidate = KNN(k=iters_match+1)
        _, nb0 = knn_sim(xy1, xy1)
        _, nb1 = knn_sim(xy2, xy2)
        _, sub_candidate = knn_candidate(xy2, xy11)

        transport_cross, similarity_cross = track_ot.sinkhorn(
            fmap1.transpose(1, -1),
            fmap2.transpose(1, -1),
            xy11,
            xy2,
            dis_range,
            epsilon=torch.exp(self.epsilon) + 0.03,
            gamma=torch.exp(self.gamma),
            max_iter=self.nb_iter,
            candidate=sub_candidate,
        )

        _, column_second_max_indices, row_second_max_indices, condition_indices = get_recon_flow(
            transport_cross, [xy1, xy2])

        track = condition_indices.int()
        idx_sub = row_second_max_indices[:, :, 0] + 1

        idx_sub = idx_sub * track - 1

        appear_mask = torch.any(sub_candidate[..., :dis_range[0]] == idx_sub.unsqueeze(2), dim=2)
        idx_sub = torch.where(appear_mask, idx_sub, torch.tensor(-1, device=xy1.device))

        idx_sub = similarity_verify([xy1, xy2], idx_sub, self.sim_neighbors, self.threshold_similarity, nb0, nb1)

        idx_sub_clean, _ = removeoutlier([xy1, xy2], idx_sub + 1, self.outlier_neighbors, self.threshold_outlier)
        valid_counts = torch.sum((idx_sub != -1), dim=-1)
        valid_mask_batch = (valid_counts > 15).unsqueeze(-1)
        idx_sub = torch.where(valid_mask_batch, idx_sub_clean, idx_sub + 1)

        a = 0
        for iter in range(1, iters_match):
            num0 = torch.sum((idx_sub != 0), dim=-1)
            dis_range[0] = dis_range[0] + dis_range[1]

            transport_cross, similarity_cross = track_ot.sinkhorn(
                fmap1.transpose(1, -1),
                fmap2.transpose(1, -1),
                xy11,
                xy2,
                dis_range,
                epsilon=torch.exp(self.epsilon) + 0.03,
                gamma=torch.exp(self.gamma),
                max_iter=self.nb_iter,
                candidate=sub_candidate,
            )

            _, column_second_max_indices, row_second_max_indices, condition_indices = get_recon_flow(
                transport_cross, p)

            track = condition_indices.int()
            idx_sub0 = row_second_max_indices[:, :, 0] + 1
            idx_sub0 = idx_sub0 * track - 1
            zero_mask = (idx_sub == 0)

            match_mask = (idx_sub != 0)
            idx_sub0[match_mask] = -1

            idx_sub = torch.where(zero_mask, idx_sub0 + 1, idx_sub)
            idx_sub = idx_sub - 1

            appear_mask = torch.any(sub_candidate[..., :dis_range[0]] == idx_sub.unsqueeze(2), dim=2)
            idx_sub = torch.where(appear_mask, idx_sub, torch.tensor(-1, device=xy1.device))

            idx_sub = similarity_verify([xy1, xy2], idx_sub, self.sim_neighbors, self.threshold_similarity, nb0, nb1)

            idx_sub_clean, _ = removeoutlier([xy1, xy2], idx_sub + 1, self.outlier_neighbors, self.threshold_outlier)
            valid_counts = torch.sum((idx_sub != -1), dim=-1)
            valid_mask_batch = (valid_counts > 15).unsqueeze(-1)
            idx_sub = torch.where(valid_mask_batch, idx_sub_clean, idx_sub + 1)

            for i in range(b):
                flattened_idx = idx_sub[i, :]
                counts = torch.bincount(flattened_idx)
                indices_to_zero = torch.nonzero(counts > 1).squeeze()
                flattened_idx = torch.where(torch.isin(flattened_idx, indices_to_zero), torch.tensor(0, device=xy1.device),
                                            flattened_idx)
                idx_sub[i, :] = flattened_idx

            num = torch.sum((idx_sub != 0), dim=-1)

            if torch.all(num - num0 <= 0):
                a = a + 1
            else:
                a = a - 1
            if a >= 1:
                break

        track_mask = (idx_sub != 0)
        idx_sub = idx_sub - 1
        idx_full = idx_sub * track_mask
        idx_full_expand = idx_full.unsqueeze(-1).expand(-1, -1, 3)

        pc_2 = torch.gather(xy2, 1, idx_full_expand)

        flow = pc_2 - xy1
        
        flow = flow * track_mask.unsqueeze(-1).float()
        
        flow_gri = griddata_flow(xy1, flow.clone(), track_mask)

        return flow, flow_gri, idx_sub

    def knn_match(self, p, flow_pred):
        [xy1, xy2] = p
        xy11 = xy1 + flow_pred
        dis_range = [1, 1]
        knn = KNN(k=12)
        _, nb0 = knn(xy1, xy1)
        _, nb1 = knn(xy2, xy2)
        _, sub_candidate = knn(xy2, xy11)
        distance = torch.cdist(xy11, xy2, p=2)

        _, column_second_min_indices, row_second_min_indices, condition_indices = get_recon_flow_knn(
            distance, [xy11, xy2])

        track = condition_indices.int()
        idx_sub = row_second_min_indices[:, :, 0] + 1

        idx_sub = idx_sub * track - 1

        appear_mask = torch.any(sub_candidate[..., :dis_range[0]] == idx_sub.unsqueeze(2), dim=2)
        
        idx_sub = torch.where(appear_mask, idx_sub, torch.tensor(-1, device=xy1.device))

        idx_sub = similarity_verify([xy1, xy2], idx_sub, self.sim_neighbors, self.threshold_similarity, nb0, nb1)

        idx_sub_clean, _ = removeoutlier([xy1, xy2], idx_sub + 1, self.outlier_neighbors, self.threshold_outlier)
        valid_counts = torch.sum((idx_sub != -1), dim=-1)
        valid_mask_batch = (valid_counts > 15).unsqueeze(-1)
        idx_sub = torch.where(valid_mask_batch, idx_sub_clean, idx_sub + 1)

        track_mask = (idx_sub != 0)
        idx_sub = idx_sub - 1
        idx_full = idx_sub * track_mask
        idx_full_expand = idx_full.unsqueeze(-1).expand(-1, -1, 3)

        pc_2 = torch.gather(xy2, 1, idx_full_expand)
        flow = pc_2 - xy1
        
        flow = flow * track_mask.unsqueeze(-1).float()
        
        flow_gri = griddata_flow(xy1, flow.clone(), track_mask)

        return flow, flow_gri, idx_sub
    def voxel_split(self, p, fmap1, fmap2, num_divs=(2, 3, 4), margin=0.1):
        xy1, xy2 = p[0], p[1]  # [1, N1, 3], [1, N2, 3]
        fmap1 = fmap1.squeeze(0)  # [f, N1]
        fmap2 = fmap2.squeeze(0)  # [f, N2]
        xy1 = xy1.squeeze(0)      # [N1, 3]
        xy2 = xy2.squeeze(0)      # [N2, 3]
    
        device = xy1.device
    
        xyz_min = xy1.min(dim=0)[0]-0.1
        xyz_max = xy1.max(dim=0)[0]+0.1
        xyz_size = xyz_max - xyz_min

        dx, dy, dz = xyz_size / torch.tensor(num_divs, device=device)
    
        pc1_split, fmap1_split, idx1_split = [], [], []
        pc2_split, fmap2_split, idx2_split = [], [], []
        for ix in range(num_divs[0]):
            for iy in range(num_divs[1]):
                for iz in range(num_divs[2]):
                    box_min = xyz_min + torch.tensor([ix*dx, iy*dy, iz*dz], device=device)
                    box_max = box_min + torch.tensor([dx, dy, dz], device=device)
    
                    mask1 = ((xy1 >= box_min) & (xy1 <= box_max)).all(dim=1)
                    idx1 = torch.nonzero(mask1).squeeze(-1)
                    if idx1.numel() == 0:
                        continue
    
                    margin_box_min = box_min - margin
                    margin_box_max = box_max + margin
                    mask2 = ((xy2 >= margin_box_min) & (xy2 <= margin_box_max)).all(dim=1)
                    idx2 = torch.nonzero(mask2).squeeze(-1)
                    if idx2.numel() == 0:
                        continue
    
                    pos1_local = xy1[idx1].unsqueeze(0)                # [M1, 3]
                    fmap1_local = fmap1[:, idx1].unsqueeze(0)          # [f, M1]
                    
                    pc1_split.append(pos1_local)
                    fmap1_split.append(fmap1_local)
                    idx1_split.append(idx1)
                    pos2_local = xy2[idx2].unsqueeze(0)                # [M2, 3]
                    fmap2_local = fmap2[:, idx2].unsqueeze(0)          # [f, M2]
                    pc2_split.append(pos2_local)
                    fmap2_split.append(fmap2_local)
                    idx2_split.append(idx2)
                    
        return pc1_split, fmap1_split, idx1_split, pc2_split, fmap2_split, idx2_split
    
    def forward(self, p, num_iters=12):
        [xy1, xy2] = p  

        fmap1 = self.feature_extractor(p[0])
        fmap2 = self.feature_extractor(p[1])

        flow0 = torch.zeros_like(xy1)

        if self.mode == 'GOTrack+':
            scale_min = xy1.amin(dim=1, keepdim=True)   
            scale_max = xy1.amax(dim=1, keepdim=True)   
            scale_range = scale_max - scale_min + 1e-6       
            
            pos1_local = (xy1 - scale_min) / scale_range * (2 * torch.pi) - torch.pi
            pos2_local = (xy2 - scale_min) / scale_range * (2 * torch.pi) - torch.pi
            
            nn_distance = KnnDistance(pos1_local, 3)
            voxel_scale = self.scale_offset * nn_distance
    
            transport = ot.sinkhorn(fmap1.transpose(1, -1), fmap2.transpose(1, -1), pos1_local, pos2_local,
                                    epsilon=torch.exp(self.epsilon) + 0.03,
                                    gamma=self.gamma,  
                                    max_iter=1)
            self.corr_block.init_module(fmap1, fmap2, pos2_local, transport)
    
            fct1, graph_context = self.context_extractor(pos1_local)
    
            net, inp = torch.split(fct1, [self.hidden_dim, self.context_dim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
    
            coords1, coords2 = pos1_local, pos1_local
            flow_predictions = []
            all_delta_flow = []
    
            for itr in range(num_iters):
                coords2 = coords2.detach()
                corr = self.corr_block(coords=coords2, all_delta_flow=all_delta_flow, num_iters=num_iters,
                                       scale=voxel_scale)
                flow = coords2 - coords1
                net, delta_flow = self.update_block(net, inp, corr, flow, graph_context)
                all_delta_flow.append(delta_flow)
                coords2 = coords2 + delta_flow
                flow_predictions.append(coords2 - coords1)
    
            flow_pred = flow_predictions[-1] * scale_range / (2 * torch.pi)  
            
            flow0 = flow_pred

        flow, flow_gri, track_id = self.matching(p, flow0, fmap1, fmap2, self.candidates)
        flow, flow_gri, track_id = self.matching(p, flow_gri, fmap1, fmap2, self.candidates)

        flow, flow_gri, track_id = self.knn_match(p, flow_gri)
        flow, flow_gri, track_id = self.knn_match(p, flow_gri)

        return flow, flow_gri, track_id
