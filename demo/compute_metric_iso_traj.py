import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
import os
import scipy.io as sio
from collections import defaultdict

ppp = '0.01'
Min_Length = 4
NumFrames = 21
Dist_Thresh = 0.0075 # ISO 0.005, ABC 0.0075

gt_mat_path = rf"test/ABC_TR_{ppp}ppp/Parameter_Particles/GT_Tracks_{ppp}ppp.mat"
# pred_mat_path = rf"test/ISO_TR_{ppp}ppp/results {ppp}ppp_NND_NNP/Tracks_{ppp}ppp.mat"

pred_mat_path = rf"test/ABC_TR_{ppp}ppp/results {ppp}ppp_NND_NNP/Tracks_{ppp}ppp.mat"

def parse_mat_to_flat(mat_path, struct_name, min_length):
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"File {mat_path} not found")
        
    flat_list = []
    valid_track_id = 0
    
    mat_data = sio.loadmat(mat_path)
    
    struct_data = mat_data[struct_name]
    struct_data = np.squeeze(struct_data)
    
    if struct_data.ndim == 0:
        struct_data = np.array([struct_data])
        
    N = len(struct_data)
    
    for i in range(N):
        item = struct_data[i]
        
        frames = np.squeeze(item['Frames'])
        coords = np.squeeze(item['Coords'])
        
        if frames.ndim == 0:
            frames = np.array([frames])
            
        L = len(frames)
        
        if L >= min_length:
            if coords.shape == (3, L) and coords.shape != (L, 3):
                coords = coords.T
            elif coords.ndim == 1 and L == 1 and len(coords) == 3:
                coords = coords.reshape(1, 3)
            elif coords.shape != (L, 3):
                coords = coords.reshape(L, 3) 

            track_ids = np.full((L, 1), valid_track_id, dtype=np.int32)
            frames_col = frames.reshape(L, 1).astype(np.int32)
            
            flat_data = np.hstack((track_ids, frames_col, coords))
            flat_list.append(flat_data)
            valid_track_id += 1
            
    if not flat_list:
        return np.empty((0, 5)), 0
        
    return np.vstack(flat_list), valid_track_id


def compute_kinematics(flat_data, num_tracks):
    V = np.full((len(flat_data), 3), np.nan)
    A = np.full((len(flat_data), 3), np.nan)
    
    track_ids = flat_data[:, 0].astype(int)
    frames = flat_data[:, 1].astype(int)
    coords = flat_data[:, 2:5]
    
    track_lengths = np.bincount(track_ids, minlength=num_tracks)
    track_ends = np.cumsum(track_lengths)
    track_starts = track_ends - track_lengths
    
    for i in range(num_tracks):
        start, end = track_starts[i], track_ends[i]
        if end - start < 3:
            continue 
            
        t_seq = frames[start:end]
        p_seq = coords[start:end]
        
        for j in range(1, len(t_seq) - 1):
            if t_seq[j] == t_seq[j-1] + 1 and t_seq[j] == t_seq[j+1] - 1:
                global_idx = start + j
                V[global_idx] = (p_seq[j+1] - p_seq[j-1]) / 2.0
                A[global_idx] = p_seq[j+1] - 2.0 * p_seq[j] + p_seq[j-1]
                
    return V, A


print("Loading and parsing .mat data...")
gt_flat_data, num_gt = parse_mat_to_flat(gt_mat_path, 'GT_Tracks_Clean', Min_Length)
pred_flat_data, num_pred = parse_mat_to_flat(pred_mat_path, 'Pred_Tracks', Min_Length)

if num_gt == 0 or num_pred == 0:
    raise ValueError("After cleaning, there is no effective trajectory, and the indicators cannot be calculated.")


gt_flat_data = np.hstack((gt_flat_data, np.arange(len(gt_flat_data)).reshape(-1, 1)))
pred_flat_data = np.hstack((pred_flat_data, np.arange(len(pred_flat_data)).reshape(-1, 1)))


print("Calculating Lagrangian quantities (V, A)...")
gt_V, gt_A = compute_kinematics(gt_flat_data, num_gt)
pred_V, pred_A = compute_kinematics(pred_flat_data, num_pred)

gt_lengths = np.bincount(gt_flat_data[:, 0].astype(int), minlength=num_gt)
pred_lengths = np.bincount(pred_flat_data[:, 0].astype(int), minlength=num_pred)
Total_GT_Points = np.sum(gt_lengths)

gt_track_frames = defaultdict(set)
for row in gt_flat_data[:, 0:2].astype(int):
    gt_track_frames[row[0]].add(row[1])

pred_track_frames = defaultdict(set)
for row in pred_flat_data[:, 0:2].astype(int):
    pred_track_frames[row[0]].add(row[1])


print("Spatiotemporal matching and point-to-point relationship establishment are underway...")
row_idx_list, col_idx_list = [], []
matched_pred_global_indices = []
matched_gt_global_indices = []
matched_pred_track_ids = [] 

Total_Match_Points = 0
Total_Pos_Error = 0.0

for t in range(NumFrames):
    gt_curr = gt_flat_data[gt_flat_data[:, 1] == t]
    pred_curr = pred_flat_data[pred_flat_data[:, 1] == t]
    
    if len(gt_curr) == 0 or len(pred_curr) == 0: continue
        
    gt_coords, gt_ids, gt_global = gt_curr[:, 2:5], gt_curr[:, 0].astype(int), gt_curr[:, 5].astype(int)
    pred_coords, pred_ids, pred_global = pred_curr[:, 2:5], pred_curr[:, 0].astype(int), pred_curr[:, 5].astype(int)
    
    dists, nn_idx = cKDTree(gt_coords).query(pred_coords, distance_upper_bound=Dist_Thresh)
    valid_mask = dists <= Dist_Thresh
    
    row_idx_list.append(pred_ids[valid_mask])
    col_idx_list.append(gt_ids[nn_idx[valid_mask]])
    
    matched_pred_global_indices.extend(pred_global[valid_mask])
    matched_gt_global_indices.extend(gt_global[nn_idx[valid_mask]])
    matched_pred_track_ids.extend(pred_ids[valid_mask]) 
    Total_Pos_Error += np.sum(dists[valid_mask])
    Total_Match_Points += np.sum(valid_mask)

if row_idx_list:
    all_rows = np.concatenate(row_idx_list)
    all_cols = np.concatenate(col_idx_list)
    all_data = np.ones(len(all_rows), dtype=np.int32)
    Vote_Matrix = sp.coo_matrix((all_data, (all_rows, all_cols)), shape=(num_pred, num_gt)).tocsr()
else:
    Vote_Matrix = sp.csr_matrix((num_pred, num_gt), dtype=np.int32)


print("Compiling evaluation indicators...")

Point_Recall = Total_Match_Points / Total_GT_Points if Total_GT_Points > 0 else 0

valid_gt_count = np.sum(gt_lengths > 0) 

max_hits = Vote_Matrix.max(axis=1).toarray().flatten()
assigned_gts = Vote_Matrix.argmax(axis=1).A1

valid_pred_mask = pred_lengths > 0
Cr = np.zeros(num_pred)
Cr[valid_pred_mask] = max_hits[valid_pred_mask] / pred_lengths[valid_pred_mask]

is_ghost = Cr < 0.5  
Ghost_Track_Count = np.sum(is_ghost[valid_pred_mask]) + np.sum(~valid_pred_mask)

GTR = Ghost_Track_Count / valid_gt_count if valid_gt_count > 0 else 0

GT_Assigned_Preds = [[] for _ in range(num_gt)]
valid_assignment_idx = np.where((~is_ghost) & valid_pred_mask)[0]
for p in valid_assignment_idx:
    GT_Assigned_Preds[assigned_gts[p]].append(p)

MT_Count, ML_Count, Total_Breaks = 0, 0, 0
Coverage_List = []

for g in range(num_gt):
    len_gt = gt_lengths[g]
    if len_gt == 0: continue
        
    assigned_preds = GT_Assigned_Preds[g]
    if not assigned_preds:
        ML_Count += 1
        Coverage_List.append(0.0)
        continue
        
    covered_frames = set()
    for p_id in assigned_preds:
        covered_frames.update(pred_track_frames[p_id])
        
    valid_covered = covered_frames.intersection(gt_track_frames[g])
    Coverage = len(valid_covered) / len_gt
    Coverage_List.append(Coverage)
    
    if Coverage >= 0.80: MT_Count += 1
    elif Coverage < 0.20: ML_Count += 1
        
    Total_Breaks += (len(assigned_preds) - 1)

valid_gt_count = np.sum(gt_lengths > 0)
MTC = np.mean(Coverage_List) if Coverage_List else 0
Ratio_MT = MT_Count / valid_gt_count if valid_gt_count else 0
Ratio_ML = ML_Count / valid_gt_count if valid_gt_count else 0
IDS_Ratio = (Total_Breaks / Total_Match_Points * 1000) if Total_Match_Points > 0 else 0

Mean_Pos_Error = (Total_Pos_Error / Total_Match_Points / Dist_Thresh) if Total_Match_Points > 0 else float('nan')

m_p_idx_raw = np.array(matched_pred_global_indices)
m_g_idx_raw = np.array(matched_gt_global_indices)
m_p_trk_raw = np.array(matched_pred_track_ids)

if len(m_p_idx_raw) > 0:
    valid_kinematic_mask = ~is_ghost[m_p_trk_raw]
    
    m_p_idx = m_p_idx_raw[valid_kinematic_mask]
    m_g_idx = m_g_idx_raw[valid_kinematic_mask]
    
    if len(m_p_idx) > 0:
        diff_V = pred_V[m_p_idx] - gt_V[m_g_idx]
        norm_V = np.linalg.norm(diff_V, axis=1)
        valid_V_mask = ~np.isnan(norm_V)
        Mean_Vel_Error = np.mean(norm_V[valid_V_mask]) if np.sum(valid_V_mask) > 0 else float('nan')
        
        diff_A = pred_A[m_p_idx] - gt_A[m_g_idx]
        norm_A = np.linalg.norm(diff_A, axis=1)
        valid_A_mask = ~np.isnan(norm_A)
        Mean_Acc_Error = np.mean(norm_A[valid_A_mask]) if np.sum(valid_A_mask) > 0 else float('nan')
    else:
        Mean_Vel_Error = float('nan')
        Mean_Acc_Error = float('nan')
else:
    Mean_Vel_Error = float('nan')
    Mean_Acc_Error = float('nan')


print(f"[Dimension 1] Data Capture Capability (Completeness & Yield)")
print(f"  1. Recall (global point recall)        : {Point_Recall * 100:.2f}%  ({Total_Match_Points}/{Total_GT_Points})")

print(f"[Dimension 2] Temporal Continuity")
print(f"  2. MTC (mean trajectory completeness)  : {MTC * 100:.2f}%")
print(f"  3. MT  (mostly tracked ratio >= 80%)   : {Ratio_MT * 100:.2f}%")
print(f"  4. ML  (mostly lost ratio < 20%)       : {Ratio_ML * 100:.2f}%")

print(f"[Dimension 3] Topological Purity")
print(f"  5. GTR (ghost trajectory ratio)        : {GTR * 100:.2f}%")
print(f"  6. IDS (identity-switch rate)          : {IDS_Ratio:.2f} (per 1000 matched points)")

print(f"[Dimension 4] Lagrangian Kinematic Fidelity")
print(f"  7. Err_Pos (position error)            : {Mean_Pos_Error:.4f} px (normalized)")
print(f"  8. Err_Vel (instantaneous velocity error): {Mean_Vel_Error/Dist_Thresh:.4f} (unit/frame)")
print(f"  9. Err_Acc (material acceleration error): {Mean_Acc_Error/Dist_Thresh:.4f} (unit/frame^2)")

print("=" * 50 + "\n")