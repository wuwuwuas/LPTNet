import torch
from scipy.interpolate import griddata, Rbf, RBFInterpolator
import numpy as np
from .KNN_torch import KNN

def similarity_verify(pclouds, track, neighbors, thres, neigh_p0, neigh_p1):
    track_mask = (track != -1)
    idx = track
    neigh_k = neighbors

    neigh_p0 = neigh_p0[..., 1:neigh_k + 1]
    neigh_p1 = neigh_p1[..., 1:neigh_k + 1]
    
    idx_safe = torch.where(track_mask, idx, torch.tensor(0, device=idx.device))

    idx_expanded = idx_safe.unsqueeze(-1).expand(-1, -1, neigh_k)
    x1n = torch.gather(neigh_p1, 1, idx_expanded)
    
    x0n_all = torch.gather(idx, 1, neigh_p0.flatten(1)).view(neigh_p0.shape)
    
    num_tensor = (x1n.unsqueeze(3) == x0n_all.unsqueeze(2)).any(dim=3).sum(dim=2)
    num = num_tensor.float() 

    sim_mask = num >= thres
    final_mask = track_mask & sim_mask
    mapped_idx = torch.where(final_mask, idx, torch.tensor(-1, device=idx.device))

    return mapped_idx

def removeoutlier(pclouds, track, neighbors, thres):
    pc_0, pc_1 = pclouds[0], pclouds[1]
    b, n, c = pc_0.shape
    
    track_mask = (track != 0)
    
    idx_safe = torch.where(track_mask, track - 1, torch.tensor(0, device=track.device))
    
    pc_1_mapped = torch.gather(pc_1, 1, idx_safe.unsqueeze(-1).expand(-1, -1, 3))
    flow = pc_1_mapped - pc_0
    
    knn = KNN(k=neighbors)
    _, neigh_p = knn(pc_0, pc_0)
    
    neigh_p_exp = neigh_p.unsqueeze(-1).expand(b, n, neighbors, c)
    flow_n = torch.gather(flow, 1, neigh_p_exp.contiguous().view(b, -1, c)).view(b, n, neighbors, c)
    
    valid_n_mask = torch.gather(track_mask, 1, neigh_p.view(b, -1)).view(b, n, neighbors)
    flow_self = flow.unsqueeze(2).expand(b, n, neighbors, c)
    flow_n = torch.where(valid_n_mask.unsqueeze(-1), flow_n, flow_self)

    flow_m, _ = torch.median(flow_n, dim=-2)
    e = 0.075
    r = torch.abs(flow_n - flow_m.unsqueeze(-2))
    r, _ = torch.median(r, dim=-2)
    r = r + e
    
    rn = torch.abs(flow - flow_m) / r
    IDX = (rn > thres).any(dim=-1)
    
    track_out = torch.where(IDX | (~track_mask), torch.tensor(0, device=track.device), track)
    
    return track_out, flow

def batched_rbf_interpolate_gpu(x_src, y_src, x_target, src_mask, smoothing=1e-5):
    src_mask_expanded = src_mask.unsqueeze(-1).float()
    x_src = x_src * src_mask_expanded
    y_src = y_src * src_mask_expanded

    A = torch.cdist(x_src, x_src)

    penalty_diag = (~src_mask).float() * 1e4 
    penalty = torch.diag_embed(penalty_diag) 

    if smoothing > 0:
        A += torch.eye(x_src.shape[1], device=x_src.device, dtype=x_src.dtype).unsqueeze(0) * smoothing

    A += penalty

    b, n, _ = A.shape
    if b <= 8 and n > 512:
        W_list = [torch.linalg.solve(A[i], y_src[i]) for i in range(b)]
        W = torch.stack(W_list, dim=0)
    else:
        W = torch.linalg.solve(A, y_src)

    B = torch.cdist(x_target, x_src)
    y_target = torch.bmm(B, W)

    return y_target
def griddata_flow(pclouds, flow, trackmask):
    valid_mask = (trackmask == 1)
    target_mask = (trackmask != 1)

    inter_flow = batched_rbf_interpolate_gpu(
        x_src=pclouds, 
        y_src=flow, 
        x_target=pclouds, 
        src_mask=valid_mask,
        smoothing=1e-5  
    )

    target_mask_expanded = target_mask.unsqueeze(-1).float()
    valid_mask_expanded = valid_mask.unsqueeze(-1).float()
    
    flow = flow * valid_mask_expanded + inter_flow * target_mask_expanded

    return flow

def get_recon_flow(transport_cross, pclouds):  
    row_second_max_values, row_second_max_indices = torch.topk(transport_cross, k=2, dim=-1)
    column_second_max_values, column_second_max_indices = torch.topk(transport_cross, k=2, dim=-2)

    row_max_indices_expanded = row_second_max_indices[:, :, 0].unsqueeze(2).expand(-1, -1, 3)
    # pc1 = torch.gather(pclouds[1], 1, row_max_indices_expanded)
    pc1 = torch.gather(pclouds[1], 1, row_max_indices_expanded).contiguous()
    flow = pc1 - pclouds[0]

    row = torch.arange(pclouds[0].shape[1], device='cuda').unsqueeze(0)
    condition_indices = row == torch.gather(column_second_max_indices[:, 0, :], 1, row_second_max_indices[:, :, 0])
    flow = flow * condition_indices.unsqueeze(-1)
    return flow, column_second_max_indices, row_second_max_indices, condition_indices

def get_recon_flow_knn(distance, pclouds):  
    row_second_min_values, row_second_min_indices = torch.topk(distance, k=2, dim=-1, largest=False)
    column_second_min_values, column_second_min_indices = torch.topk(distance, k=2, dim=-2, largest=False)

    row_min_indices_expanded = row_second_min_indices[:, :, 0].unsqueeze(2).expand(-1, -1, 3)
    pc1 = torch.gather(pclouds[1], 1, row_min_indices_expanded)
    flow = pc1 - pclouds[0]

    row = torch.arange(pclouds[0].shape[1], device='cuda').unsqueeze(0)
    condition_indices = row == torch.gather(column_second_min_indices[:, 0, :], 1, row_second_min_indices[:, :, 0])

    flow = flow * condition_indices.unsqueeze(-1)
    return flow, column_second_min_indices, row_second_min_indices, condition_indices