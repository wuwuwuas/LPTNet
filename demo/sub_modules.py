import os
import numpy as np
import torch
import pyopenlpt as lpt 
import itertools
import sys
import torch.nn.functional as F
from scipy.spatial import cKDTree
import csv
import pandas as pd
import math


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w', encoding='utf-8') 

    def write(self, message):
        self.terminal.write(message) 
        self.log.write(message)      
        self.log.flush()             

    def flush(self):
        pass

def tracker_STB(pts_p, pts_q, search_radius=2.0):
    if len(pts_p) == 0 or len(pts_q) == 0:
        return np.full(len(pts_p), -1, dtype=int)
    tree = cKDTree(pts_q)
    dists, idxs = tree.query(pts_p, k=1, distance_upper_bound=search_radius)
    
    match_idx = np.full(len(pts_p), -1, dtype=int)
    valid = (dists <= search_radius) & (idxs < len(pts_q))
    match_idx[valid] = idxs[valid]
    return match_idx


def build_tracker_NN(tracker, nb_iter=30):
    device = next(tracker.parameters()).device
    TARGET_CORE_SIZE = 5000  
    OVERLAP = 400  
    MAX_BATCH_SIZE = 5
    def tracker_NN(pts_p, pts_q, search_radius=None):
        # 强制指定切片主轴 (0: X轴, 1: Y轴, 2: Z轴)。设为 None 则自动寻找最大跨度轴。
        debug_axis = None 
        
        if len(pts_p) == 0 or len(pts_q) == 0:
            return np.full(len(pts_p), -1, dtype=int)
            
        n_p = len(pts_p)
        n_q = len(pts_q)
        full_match_idx = np.full(n_p, -1, dtype=int)

        with torch.no_grad():
            t_p = torch.tensor(pts_p, dtype=torch.float32, device=device)
            t_q = torch.tensor(pts_q, dtype=torch.float32, device=device)

            if debug_axis is not None:
                main_axis = debug_axis
            else:
                extents = t_p.amax(dim=0) - t_p.amin(dim=0)
                main_axis = extents.argmax().item()

            _, idx_p = torch.sort(t_p[:, main_axis])
            _, idx_q = torch.sort(t_q[:, main_axis])
            
            t_p_sorted = t_p[idx_p]
            t_q_sorted = t_q[idx_q]

            K = max(1, round((n_p - 2 * OVERLAP) / TARGET_CORE_SIZE))

            if K == 1:
                M_p = n_p
                starts_p = [0]
            else:
                M_p = math.ceil((n_p + 2 * (K - 1) * OVERLAP) / K)
                S = M_p - 2 * OVERLAP 
                
                starts_p = [i * S for i in range(K - 1)]
                starts_p.append(n_p - M_p) 
            
            M_q = min(n_q, M_p + 200)

            starts_q = []
            q_axis_vals = t_q_sorted[:, main_axis].contiguous()
            for sp in starts_p:
                val = t_p_sorted[sp, main_axis]
                sq = torch.searchsorted(q_axis_vals, val).item()
                sq = max(0, min(sq, n_q - M_q))
                starts_q.append(sq)

            batch_p = torch.stack([t_p_sorted[sp : sp + M_p] for sp in starts_p])
            batch_q = torch.stack([t_q_sorted[sq : sq + M_q] for sq in starts_q])

            min_val = batch_p.amin(dim=1, keepdim=True)
            max_val = batch_p.amax(dim=1, keepdim=True)
            scale = max_val - min_val
            scale[scale == 0] = 1.0

            pc1 = ((batch_p - min_val) / scale) * 6.28 - 3.14
            pc2 = ((batch_q - min_val) / scale) * 6.28 - 3.14

            all_track_ids = []

            for i in range(0, K, MAX_BATCH_SIZE):
                sub_pc1 = pc1[i : i + MAX_BATCH_SIZE]
                sub_pc2 = pc2[i : i + MAX_BATCH_SIZE]
                
                _, _, sub_track_id = tracker([sub_pc1, sub_pc2], nb_iter)
                all_track_ids.append(sub_track_id)

            track_id = torch.cat(all_track_ids, dim=0)

            for k in range(K):
                if K == 1:
                    valid_p_local = slice(0, M_p)
                elif k == 0:
                    valid_p_local = slice(0, M_p - OVERLAP)
                elif k == K - 1:
                    prev_end = starts_p[k-1] + M_p - OVERLAP
                    local_start = max(0, prev_end - starts_p[k])
                    valid_p_local = slice(local_start, M_p)
                else:
                    valid_p_local = slice(OVERLAP, M_p - OVERLAP)

                valid_local_idx = torch.arange(valid_p_local.start, valid_p_local.stop, device=device)
                local_q_matches = track_id[k, valid_local_idx]

                valid_mask = (local_q_matches != -1) & (local_q_matches < M_q)
                if not valid_mask.any():
                    continue

                valid_local_p = valid_local_idx[valid_mask]
                valid_local_q = local_q_matches[valid_mask]

                global_p = idx_p[starts_p[k] + valid_local_p]
                global_q = idx_q[starts_q[k] + valid_local_q]

                full_match_idx[global_p.cpu().numpy()] = global_q.cpu().numpy()

        return full_match_idx

    return tracker_NN
def detect_particles_STB(lpt_img_list, active_cam_ids, tr_cfg):
    num_cams = len(lpt_img_list)
    finder = lpt.ObjectFinder2D()
    obj2d_lists_cpp = [[] for _ in range(num_cams)]
    for cam_id in active_cam_ids:
        obj2d_lists_cpp[cam_id] = finder.findObject2D(lpt_img_list[cam_id], tr_cfg)
    return obj2d_lists_cpp

def extract_rough_centers(img_tensor, threshold=0.2, kernel_size=3, min_energy_ratio=1.5):
    max_pooled = F.max_pool2d(img_tensor, kernel_size=kernel_size, stride=1, padding=1)
    peak_mask = (img_tensor == max_pooled) & (img_tensor >= threshold)
    if peak_mask.dim() == 4:
        peak_mask[:, :, 0, :] = False
        peak_mask[:, :, -1, :] = False
        peak_mask[:, :, :, 0] = False
        peak_mask[:, :, :, -1] = False
    elif peak_mask.dim() == 2:
        peak_mask[0, :] = False
        peak_mask[-1, :] = False
        peak_mask[:, 0] = False
        peak_mask[:, -1] = False
    return peak_mask.half()

def detect_particles_NN(finder, model_S1, model_S2, lpt_img_list, active_cam_ids, cam_res, use_s1=True, OTF=True):
    device = next(model_S1.parameters()).device
    device_type = 'cuda' if device.type == 'cuda' else 'cpu' 
    
    patch_size = 7
    half = patch_size // 2
    H, W = cam_res 
    
    active_cams = sorted(list(active_cam_ids))
    num_total_cams = len(lpt_img_list)
    obj2d_lists_cpp = [[] for _ in range(num_total_cams)]
    
    if not active_cams:
        return obj2d_lists_cpp
        
    B = len(active_cams)
    
    batched_imgs_np = np.empty((B, 1, H, W), dtype=np.uint8)
    for i, cam_id in enumerate(active_cams):
        img_flat = np.array(lpt_img_list[cam_id], dtype=np.uint8) if hasattr(lpt_img_list[cam_id], '__array_interface__') else np.array(lpt_img_list[cam_id].to_list(), dtype=np.uint8)
        batched_imgs_np[i, 0, :, :] = img_flat.reshape(H, W)
        
    img_tensor_batched = torch.from_numpy(batched_imgs_np).to(device, non_blocking=True)
    img_tensor_batched = img_tensor_batched.float() / 255.0

    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.float16):
        if use_s1:
            first_logits = model_S1(img_tensor_batched)
            s1_prob_batched = first_logits
        else:
            s1_prob_batched = extract_rough_centers(img_tensor_batched, threshold=0.1)
            
        combined_map_batched = img_tensor_batched

        for i, cam_id in enumerate(active_cams):
            s1_prob = s1_prob_batched[i, 0] 
            
            ys, xs = torch.where(s1_prob > 0.5) 
            N = ys.size(0)
            
            if N > 0:
                combined_map = combined_map_batched[i] 
                
                offset = torch.arange(-half, half + 1, device=device, dtype=torch.float16)
                offset_y = offset.view(1, patch_size, 1).expand(N, patch_size, patch_size)
                offset_x = offset.view(1, 1, patch_size).expand(N, patch_size, patch_size)
                
                grid_y = ys.view(N, 1, 1).half() + offset_y
                grid_x = xs.view(N, 1, 1).half() + offset_x
                
                norm_x = ((grid_x.float() / (W - 1)) * 2.0 - 1.0)
                norm_y = ((grid_y.float() / (H - 1)) * 2.0 - 1.0)
                
                grid = torch.stack((norm_x, norm_y), dim=-1)
                
                img_expanded = combined_map.unsqueeze(0).expand(N, -1, -1, -1)
                
                patches = F.grid_sample(img_expanded, grid, align_corners=True, padding_mode='zeros')
                
                if N > 1500000:
                    s2_outs_list = []
                    for batch_patches in torch.split(patches, 100000):
                        s2_outs_list.append(model_S2(batch_patches))
                    s2_outs = torch.cat(s2_outs_list, dim=0).float()
                else:
                    s2_outs = model_S2(patches).float() 
                
                dy = s2_outs[..., 0]  # Shape: (N, K)
                dx = s2_outs[..., 1]  # Shape: (N, K)
                log_I = s2_outs[..., 2] 
                log_d = s2_outs[..., 3] 
                conf = s2_outs[..., 4] # Shape: (N, K)
                
                mask = (conf > 0.1)

                real_y = ys.unsqueeze(1).float() + dy
                real_x = xs.unsqueeze(1).float() + dx

                valid_y = real_y[mask]
                valid_x = real_x[mask]
                valid_logd = log_d[mask]
                valid_logI = log_I[mask]
                if valid_x.numel() > 0:
                    valid_x_np = valid_x.cpu().numpy()
                    valid_y_np = valid_y.cpu().numpy()
                    diameters_np = np.exp(valid_logd.cpu().numpy())
                    intensities_np = np.exp(valid_logI.cpu().numpy())
                    
                    mask = (diameters_np >= 1) & (intensities_np >= 0.1)
                    valid_x_np, valid_y_np = valid_x_np[mask], valid_y_np[mask]
                    diameters_np, intensities_np = diameters_np[mask], intensities_np[mask]

                    centers_c = np.ascontiguousarray(np.column_stack((valid_x_np, valid_y_np)), dtype=np.float64)
                    diameters_c = np.ascontiguousarray(diameters_np, dtype=np.float64)
                    if OTF:
                        intensities_c = np.zeros_like(intensities_np, dtype=np.float64)
                    else:
                        intensities_c = np.ascontiguousarray(intensities_np, dtype=np.float64)
                    
                    obj2d_lists_cpp[cam_id] = finder.findTracer2D_fromNN(centers_c, diameters_c, intensities_c)
    return obj2d_lists_cpp

def run_ipr_on_frame(finder, model_S1, model_S2, lpt_stb_instance, tr_cfg, current_image_list, num_cams, phase, OTF):
    ipr_param = tr_cfg._ipr_param
    orig_tol = tr_cfg._sm_param.tol_2d_px
    orig_match_count = tr_cfg._sm_param.match_cam_count
    
    all_cam_ids = list(range(num_cams))

    strategies = []
    all_cam_ids = list(range(num_cams))
    strategies.append({
        'name': "Full Cameras",
        'cams': all_cam_ids,
        'loops': ipr_param.n_loop_ipr
    })
    # 2. Reduced cameras (Drop 1..n_reduced)
    max_drop = min(ipr_param.n_cam_reduced, max(0, num_cams - 2))
    for drop in range(1, max_drop + 1):
        k = num_cams - drop
        for subset in itertools.combinations(all_cam_ids, k):
            strategies.append({
                'name': f"Reduced {k} Cams",
                'cams': list(subset),
                'loops': ipr_param.n_loop_ipr_reduced
            })
    
    results_this_frame = []
    cam_list = lpt_stb_instance._basic_setting._cam_list # 通过 helper 获取
    H = cam_list[0].getNRow()
    W = cam_list[0].getNCol()

    for strat_idx, strat in enumerate(strategies):
        active_ids = strat['cams']
        n_loops = strat['loops']
        
        for i in range(num_cams):
             cam_list[i]._is_active = (i in active_ids)
        tr_cfg._sm_param.match_cam_count = min(4, len(active_ids))

        base_tol = orig_tol
        for loop_i in range(n_loops):
            tr_cfg._sm_param.tol_2d_px = base_tol * (1.5 ** loop_i)
            print(f"Strategy {strat['name']}: Cams {active_ids} lOOP {loop_i}")
            use_s1 = (phase == 'init') and (strat['name'] == "Full Cameras") and (loop_i == 0)
            # use_s1 = True
            # A. Detect
            if model_S1 is not None and model_S2 is not None:
                obj2d_lists = detect_particles_NN(finder, model_S1, model_S2, current_image_list, active_ids, (H, W), use_s1, OTF)
            else:
                obj2d_lists = detect_particles_STB(current_image_list, active_ids, tr_cfg)
            if sum(len(x) for x in obj2d_lists) < 2: continue
            print('      found 2D objects: ', len(obj2d_lists[0]), len(obj2d_lists[1]), len(obj2d_lists[2]), len(obj2d_lists[3]))
            
            valid_objs, current_image_list = lpt.match_shake_and_calc_residual(
                cam_list, obj2d_lists, tr_cfg, current_image_list)
            
            if not valid_objs: continue
            
            results_this_frame.extend(valid_objs)
            print(f'      Matched and shaked 3D objects: {len(valid_objs)}')

            # matcher = lpt.StereoMatch(cam_list, obj2d_lists, tr_cfg)
            # candidates = matcher.match() # list[Object3D]
            # if not candidates: continue
            # # C. Shake (Optional / Lightweight) & Filter
            # shaker = lpt.Shake(cam_list, tr_cfg)
            # flags, obj3d_shaked = shaker.runShake(candidates, current_image_list)
            # valid_objs = []
            # MASK_DROP = int(lpt.ObjFlag.Ghost) | int(lpt.ObjFlag.Repeated)
            # for obj, flag in zip(obj3d_shaked, flags):
            #     if (int(flag) & MASK_DROP) == 0:
            #         valid_objs.append(obj)
            # if not valid_objs: continue
            # results_this_frame.extend(valid_objs)
            # print('      Matched 3D objects: ', len(candidates), 'After shaking: ', len(valid_objs))
            # # D. Residue Update
            # current_image_list = shaker.calResidualImage(valid_objs, current_image_list, flags=None)
            
    tr_cfg._sm_param.tol_2d_px = orig_tol
    tr_cfg._sm_param.match_cam_count = orig_match_count

    return results_this_frame, current_image_list


def save_tracks(n_cam, tracks, folder, filename):
    if not tracks:
        return
        
    filename = filename.rsplit('.', 1)[0] + '.parquet'  
    # filename = filename.rsplit('.', 1)[0] + '.csv'     
    
    filepath = os.path.join(folder, filename)
    is_parquet = filename.lower().endswith('.parquet')

    headers = ["TrackID", "FrameID", "WorldX", "WorldY", "WorldZ"]
    for i in range(n_cam):
        headers.extend([f"cam{i}_x(col)", f"cam{i}_y(row)"])

    def _extract_rows():
        empty_cam = [0.0, 0.0] * n_cam
        for tid, tr in enumerate(tracks):
            objs = tr.getObjects() if hasattr(tr, "getObjects") else getattr(tr, "_obj3d_list", [])
            frames = tr.getFrames() if hasattr(tr, "getFrames") else getattr(tr, "_t_list", [])
            
            if not objs or len(objs) == 0:
                continue
                
            first_obj = objs[0]
            is_pt_center = hasattr(first_obj, "_pt_center")
            has_cam_proj = hasattr(first_obj, "cam_proj")
            
            for i, obj in enumerate(objs):
                fid = frames[i] if i < len(frames) else -1
                
                if is_pt_center:
                    pt = obj._pt_center
                    x, y, z = pt[0], pt[1], pt[2]
                else:
                    x, y, z = obj.x, obj.y, obj.z
                    
                row_data = [tid, fid, round(x, 6), round(y, 6), round(z, 6)]
                
                if has_cam_proj:
                    projs = obj.cam_proj
                    len_projs = len(projs)
                    for c in range(n_cam):
                        if c < len_projs:
                            p2d = projs[c]
                            if isinstance(p2d, (list, tuple)):
                                row_data.extend([round(p2d[0], 4), round(p2d[1], 4)])
                            else:
                                row_data.extend([round(p2d.x, 4), round(p2d.y, 4)])
                        else:
                            row_data.extend([0.0, 0.0])
                else:
                    row_data.extend(empty_cam)
                    
                yield row_data

    try:
        if is_parquet:
            all_data = list(_extract_rows())
            if all_data:
                try:
                    df = pd.DataFrame(all_data, columns=headers)
                    df.to_parquet(filepath, engine='pyarrow', compression='snappy')
                except ImportError:
                    print("Warning: The pyarrow engine is missing; it will automatically be saved as a CSV file...")
                    filepath = filepath.rsplit('.', 1)[0] + '.csv'
                    with open(filepath, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(headers)
                        writer.writerows(all_data) 
        else:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                
                batch_data = []
                for row in _extract_rows():
                    batch_data.append(row)
                    if len(batch_data) >= 50000:
                        writer.writerows(batch_data)
                        batch_data.clear()
                        
                if batch_data:
                    writer.writerows(batch_data)
                    
    except Exception as e:
        print(f"Error writing track file {filename}: {e}")
        import traceback
        traceback.print_exc()
