import os
import sys
import copy
import numpy as np
import torch
import time
from collections import deque
import pyopenlpt as lpt 
from sub_modules import run_ipr_on_frame, save_tracks
from scipy.spatial import cKDTree
from itertools import compress
import math
import glob

class PythonSTB:
    def __init__(self, config_path, finder, model_S1, model_S2, model_tracker, predictor, n_segments=1, OTF=True):
        self.basic_settings = lpt.BasicSetting()
        self.basic_settings.readConfig(config_path)
        
        self.tr_cfg = lpt.TracerConfig()
        self.tr_cfg.readConfig(self.basic_settings._object_config_paths[0], self.basic_settings)

        self.tr_cfg._sm_param.n_segments = n_segments

        self.cpp_stb = lpt.STB(self.basic_settings, "Tracer", self.basic_settings._object_config_paths[0])
        
        self.short_tracks_active = deque()
        self.long_tracks_active = deque()
        self.long_tracks_inactive = deque()
        self.exit_tracks = deque()

        self.init_buffers = {} 
        self.n_init_frames = self.tr_cfg._stb_param._n_initial_frames
        
        self.imgio_list = []
        for path in self.basic_settings._image_file_paths:
            io = lpt.ImageIO()
            io.loadImgPath("", path)
            self.imgio_list.append(io)
        self.num_cams = len(self.imgio_list)
        
        self.finder = finder
        self.model_S1 = model_S1
        self.model_S2 = model_S2
        self.model_tracker = model_tracker
        self.predictor = predictor
        self.OTF = OTF
        self.LEN_LONG_TRACK = getattr(lpt, 'LEN_LONG_TRACK', 15)
        print(f"Python STB Initialized. Init Frames: {self.n_init_frames}")
    def run(self):
        start_frame = self.basic_settings._frame_start
        end_frame = self.basic_settings._frame_end
        os.makedirs(self.basic_settings._output_path + "ConvergeTrack/", exist_ok=True)
        total_start = time.time()
        
        for frame_id in range(start_frame, end_frame + 1):
            iter_start = time.time()
            print(f"\n>>> Processing Frame {frame_id}")
            
            image_list = [io.loadImg(frame_id) for io in self.imgio_list]
            # img_orig = [io.loadImg(frame_id) for io in self.imgio_list] # 备份原图用于 VSC (如需)
            
            frames_processed = frame_id - start_frame 
            if frames_processed < self.n_init_frames:
                self.run_init_phase_collect(frame_id, image_list)
                if frames_processed == self.n_init_frames - 1:
                    self.run_init_phase_build(start_frame)
                    self.save_all_tracks(frame_id, is_initial_phase=True, clear_history=False)
            else:
                self.run_conv_phase(frame_id, image_list)
                self.save_all_tracks(frame_id, is_initial_phase=False, clear_history=False)
                # if frame_id % 500 == 0 or frame_id == end_frame:
                #     print(f"  [Auto-Save] Checkpoint at frame {frame_id}")
                #     self.save_all_tracks(frame_id, is_initial_phase=False, clear_history=True)
            print(f"    Frame Time: {time.time() - iter_start:.4f}s")
        self.save_all_tracks(end_frame, is_initial_phase=False, clear_history=False)
        print(f"Total processing time: {time.time() - total_start:.2f}s")

    def run_init_phase_collect(self, frame_id, image_list):
        print("  [Phase] Init Collection")
        phase = 'init'
        objs, _ = run_ipr_on_frame(self.finder, self.model_S1, self.model_S2, self.cpp_stb, self.tr_cfg, image_list, self.num_cams, phase, self.OTF)
        self.init_buffers[frame_id] = objs
        print(f"  -> Found {len(objs)} particles.")

    def run_init_phase_build(self, start_frame):
        print(f"\n  [Phase] Building Initial Tracks over {self.n_init_frames} frames...")
        print(f"    Initializing tracks at Frame {start_frame}...")
        objs_f0 = self.init_buffers[start_frame]
        
        self.short_tracks_active.clear() 
        self.long_tracks_active.clear()
        
        for obj in objs_f0:
            obj._is_tracked = True 
            tr = lpt.Track()
            obj.projectObject2D(self.basic_settings._cam_list)
            tr.add_next(obj, start_frame)
            self.short_tracks_active.append(tr)

        for i in range(1, self.n_init_frames):
            prev_fid = start_frame + i - 1
            curr_fid = start_frame + i
            
            objs_prev_all = self.init_buffers[prev_fid]
            objs_curr_all = self.init_buffers[curr_fid]
            
            # --- A. 计算 PredField (C++ Logic) ---
            pf = None
            if len(objs_prev_all) > 10 and len(objs_curr_all) > 10:
                try:
                    pts_prev_vec = []
                    for o in objs_prev_all:
                        c = o._pt_center
                        pts_prev_vec.append(lpt.Pt3D(c[0], c[1], c[2]))
                        
                    pts_curr_vec = []
                    for o in objs_curr_all:
                        c = o._pt_center
                        pts_curr_vec.append(lpt.Pt3D(c[0], c[1], c[2]))
                        
                    pf = lpt.PredField(self.tr_cfg)
                    pf.calPredField_from_points(pts_prev_vec, pts_curr_vec)
                except Exception as e:
                    print(f"    [Warning] PredField error {e}")
                    pf = None

            # --- B. 准备数据供 Tracker 使用 ---
            pts_p_list = []
            if self.short_tracks_active:
                for tr in self.short_tracks_active:
                    last_obj = tr._obj3d_list[-1]
                    c = last_obj._pt_center
                    p_curr = lpt.Pt3D(c[0], c[1], c[2])
                    
                    disp = (0.0, 0.0, 0.0)
                    if pf is not None:
                        disp = pf.getDisp(p_curr)

                    pts_p_list.append([c[0] + disp[0], c[1] + disp[1], c[2] + disp[2]])
            
            pts_p = np.array(pts_p_list, dtype=float) if pts_p_list else np.empty((0, 3), dtype=float)

            # 2. 准备 pts_q (Candidates in current frame)
            if objs_curr_all:
                pts_q = np.array([[o._pt_center[0], o._pt_center[1], o._pt_center[2]] for o in objs_curr_all], dtype=float)
            else:
                pts_q = np.empty((0, 3), dtype=float)

            # 初始化当前帧 flag
            for obj in objs_curr_all:
                obj._is_tracked = False

            # --- C. 调用 Tracker (批量匹配) ---
            search_radius = self.tr_cfg._stb_param._radius_search_obj
            
            if len(pts_p) > 0 and len(pts_q) > 0:
                link_ids = self.model_tracker(pts_p, pts_q, search_radius)
            else:
                link_ids = np.full(len(self.short_tracks_active), -1, dtype=int)

            num_matches = np.sum(link_ids != -1)
            print(f"Frame {prev_fid} -> {curr_fid} | Particles: {len(pts_p)} -> {len(pts_q)} | Tracker connections: {num_matches}")

            # --- D. 更新轨迹状态 (Strict Logic) ---
            new_short_tracks = []
            
            for j, tr in enumerate(self.short_tracks_active):
                obj_idx = link_ids[j]
                
                if obj_idx != -1: 
                    target_obj = objs_curr_all[obj_idx]
                    target_obj._is_tracked = True
                    target_obj.projectObject2D(self.basic_settings._cam_list)
                    tr.add_next(target_obj, curr_fid)
                    new_short_tracks.append(tr)

            # --- E. 处理未追踪粒子 (New Tracks) ---
            for obj in objs_curr_all:
                if not obj._is_tracked:
                    obj._is_tracked = True
                    new_tr = lpt.Track()
                    obj.projectObject2D(self.basic_settings._cam_list)
                    new_tr.add_next(obj, curr_fid)
                    new_short_tracks.append(new_tr)

            self.short_tracks_active = new_short_tracks
            # self.save_all_tracks(i, is_initial_phase=True, clear_history=False)
            
        # 3. 阶段结束 晋升 Long Tracks
        print("    Promoting valid tracks...")
        final_shorts = deque()
        promoted_count = 0
        required_len = self.n_init_frames
        
        for tr in self.short_tracks_active:
            if len(tr._obj3d_list) >= required_len:
                self.long_tracks_active.append(tr)
                promoted_count += 1
            else:
                final_shorts.append(tr)
                
        self.short_tracks_active = final_shorts
        print(f"    [Init Done] Promoted to Long: {promoted_count}, Remaining Short: {len(self.short_tracks_active)}")

    def _findNN_py(self, tree, pt3d_est, radius):
        
        dist, idx = tree.query(pt3d_est, k=1, distance_upper_bound=radius)
        
        if dist == float('inf'):
            return -1
        
        if dist <= radius:
            return idx
        else:
            return -1

    def _batch_predict_tracks(self, tracks):
        import numpy as np
        n_tracks = len(tracks)
        results = [None] * n_tracks
        
        # 将轨迹按 Wiener 滤波器的阶数 (2, 3, 4, 5) 进行分组
        groups = {2: [], 3: [], 4: [], 5: []}
        
        for i, tr in enumerate(tracks):
            objs = tr._obj3d_list
            n = len(objs)
            if n < 3:
                continue
            order = n - 1 if n < 6 else 5
            start_idx = n - 1 - order
            pts = objs[start_idx:]
            
            # 提取坐标和最后一个对象（为了继承 intensity）
            coords = [[o._pt_center[0], o._pt_center[1], o._pt_center[2]] for o in pts]
            groups[order].append((i, coords, objs[-1]))
            
        kShift = 10.0
        SMALLNUMBER = 1e-6
        WIENER_MAX_ITER = 3
        
        # 按阶数批量执行底层矩阵运算
        for order, items in groups.items():
            if not items:
                continue
            
            indices = [item[0] for item in items]
            series_batch = np.array([item[1] for item in items], dtype=np.float64)
            last_objs = [item[2] for item in items]
            
            N = len(indices)
            pred_pos = np.zeros((N, 3), dtype=np.float64)
            
            for axis in range(3):
                s = series_batch[:, :, axis].copy()
                shifted = np.abs(s[:, order]) < 1.0
                s[shifted, :] += kShift
                
                denom = np.sum(s[:, :order]**2, axis=1)
                step = np.zeros(N, dtype=np.float64)
                mask_step = denom > 1e-12
                step[mask_step] = 1.0 / denom[mask_step]
                
                filt = np.zeros((N, order), dtype=np.float64)
                prediction = np.zeros(N, dtype=np.float64)
                error = s[:, order].copy()
                
                active = step > 0.0
                for _ in range(WIENER_MAX_ITER):
                    iter_active = active & (np.abs(error) > SMALLNUMBER)
                    if not np.any(iter_active):
                        break
                        
                    prediction[iter_active] = 0.0
                    for j in range(order):
                        filt[iter_active, j] += step[iter_active] * s[iter_active, j] * error[iter_active]
                        prediction[iter_active] += filt[iter_active, j] * s[iter_active, j]
                        
                    error[iter_active] = s[iter_active, order] - prediction[iter_active]
                    
                val = np.zeros(N, dtype=np.float64)
                step_zero = (step == 0.0)
                
                if order >= 1:
                    val[step_zero] = s[step_zero, order] + (s[step_zero, order] - s[step_zero, order-1])
                    
                step_pos = ~step_zero
                for j in range(order):
                    val[step_pos] += filt[step_pos, j] * s[step_pos, j+1]
                    
                val[shifted] -= kShift
                pred_pos[:, axis] = val
                
            # 检查 NaN 并批量生成 Tracer3D 对象
            valid_mask = ~np.isnan(pred_pos).any(axis=1)
            
            for i_local, idx in enumerate(indices):
                if valid_mask[i_local]:
                    try:
                        new_obj = lpt.Tracer3D()
                        # 强制转换为 float，确保 C++ 端安全接收
                        new_obj._pt_center = lpt.Pt3D(
                            float(pred_pos[i_local, 0]), 
                            float(pred_pos[i_local, 1]), 
                            float(pred_pos[i_local, 2])
                        )
                        if hasattr(last_objs[i_local], '_intensity'):
                            new_obj._intensity = last_objs[i_local]._intensity
                        results[idx] = new_obj
                    except Exception:
                        pass
                        
        return results

    # checkRepeat
    def _check_repeat_py(self, candidates, long_tracks):
        if not candidates or not long_tracks:
            return candidates
            
        # Optimization 1 unroll the inner loop and specify dtype explicitly
        long_tips = np.array([
            [tr._obj3d_list[-1]._pt_center[0], 
            tr._obj3d_list[-1]._pt_center[1], 
            tr._obj3d_list[-1]._pt_center[2]] 
            for tr in long_tracks
        ], dtype=np.float64)
        
        cand_pts = np.array([
            [o._pt_center[0], 
            o._pt_center[1], 
            o._pt_center[2]] 
            for o in candidates
        ], dtype=np.float64)
        
        tol = self.tr_cfg._sm_param.tol_3d_mm
        
        tree = cKDTree(long_tips)
        
        # Optimization 2 use query(k=1) instead of query_ball_point
        # scipy cKDTree returns np.inf for distances beyond the upper bound
        dists, _ = tree.query(cand_pts, k=1, distance_upper_bound=tol, workers=-1)
        
        valid_mask = dists == np.inf
        
        # Optimization 3 use itertools.compress for fast list filtering
        return list(compress(candidates, valid_mask))

    def _link_phase_py(self, short_tracks, long_tracks, candidates):
        n_short = len(short_tracks)
        n_cand = len(candidates)
        
        if n_cand == 0 or n_short == 0:
            return np.full(n_short, -1, dtype=np.int32).tolist()

        base_r = self.tr_cfg._stb_param._radius_search_track 

        short_p_lasts = np.array([[tr._obj3d_list[-1]._pt_center[0], 
                                   tr._obj3d_list[-1]._pt_center[1], 
                                   tr._obj3d_list[-1]._pt_center[2]] for tr in short_tracks], dtype=np.float64)
                                   
        cand_pts = np.array([[c._pt_center[0], 
                              c._pt_center[1], 
                              c._pt_center[2]] for c in candidates], dtype=np.float64)
        
        tree_cand = cKDTree(cand_pts)
        
        valid_longs = [tr for tr in long_tracks if len(tr._obj3d_list) >= 2]
        has_long = len(valid_longs) > 0
        
        if has_long:
            long_tips = np.array([[tr._obj3d_list[-1]._pt_center[0], 
                                   tr._obj3d_list[-1]._pt_center[1], 
                                   tr._obj3d_list[-1]._pt_center[2]] for tr in valid_longs], dtype=np.float64)
                                   
            long_prevs = np.array([[tr._obj3d_list[-2]._pt_center[0], 
                                    tr._obj3d_list[-2]._pt_center[1], 
                                    tr._obj3d_list[-2]._pt_center[2]] for tr in valid_longs], dtype=np.float64)
                                    
            long_disps = long_tips - long_prevs
            tree_long = cKDTree(long_tips)
            
        results = np.full(n_short, -1, dtype=np.int32)
        active_mask = np.ones(n_short, dtype=bool) 
        
        factor = 1.0
        grow = 1.1
        
        for step in range(5):
            if step > 0: 
                factor *= grow
                
            active_idx = np.where(active_mask)[0]
            if len(active_idx) == 0:
                break 
                
            curr_p_lasts = short_p_lasts[active_idx]
            n_active = len(active_idx)
            vels = np.zeros((n_active, 3), dtype=np.float64)
            
            # 提取标量运算到循环外部
            R_curr = base_r * factor
            L2_min = (0.1 * R_curr)**2
            L2_max = (0.8 * R_curr)**2
            
            if has_long:
                r_track = 3.0 * R_curr
                # 1. 改用 K-NN 搜索 (限制最多寻找 15 个最近邻)
                k_neighbors = min(50, len(long_tips))
                dists, idxs = tree_long.query(curr_p_lasts, k=k_neighbors, workers=-1)

                if k_neighbors == 1:
                    dists = dists[:, np.newaxis]
                    idxs = idxs[:, np.newaxis]
                
                # 2. 构造有效点掩码 (屏蔽距离大于 r_track 的点)
                valid_mask = dists < r_track # Shape: (n_active, k_neighbors)
                valid_counts = np.sum(valid_mask, axis=1) # 每个点实际找到的有效邻居数
                has_neighbors = valid_counts > 0 # (n_active,)
                
                if np.any(has_neighbors):
                    d2 = dists ** 2
                    d2_masked = d2 * valid_mask
                    
                    mean_d2 = np.zeros(n_active, dtype=np.float64)
                    mean_d2[has_neighbors] = np.sum(d2_masked[has_neighbors], axis=1) / valid_counts[has_neighbors]
                    
                    # 4. 矩阵化计算 L2 和 beta2
                    L2 = np.clip(0.64 * mean_d2, L2_min, L2_max)
                    beta2 = np.zeros(n_active, dtype=np.float64)
                    beta2[has_neighbors] = 1.0 / (L2[has_neighbors] + 1e-12)
                    
                    # 5. 矩阵化计算权重 W
                    w = np.zeros_like(d2)
                    w[has_neighbors] = 1.0 / (1.0 + d2[has_neighbors] * beta2[has_neighbors, np.newaxis])
                    w = w * valid_mask # 强行将超出半径的权重置零
                    
                    long_disps_k = long_disps[idxs] # Shape: (n_active, k_neighbors, 3)
                    w_sum = np.sum(w, axis=1, keepdims=True) # (n_active, 1)
                    
                    valid_pts = w_sum.squeeze(-1) > 0
                    
                    vels[valid_pts] = np.sum(
                        w[valid_pts, :, np.newaxis] * long_disps_k[valid_pts], 
                        axis=1
                    ) / w_sum[valid_pts]
            est_pos = curr_p_lasts + vels
            r_obj = base_r * factor
            
            d_cand, idx_cand = tree_cand.query(est_pos, k=1, distance_upper_bound=r_obj, workers=-1)
            
            found_mask = idx_cand < n_cand
            
            if np.any(found_mask):
                found_global_idx = active_idx[found_mask]
                results[found_global_idx] = idx_cand[found_mask]
                active_mask[found_global_idx] = False

        return results.tolist()
        
    def run_conv_phase(self, frame_id, image_list):
        phase = 'conv'
        print(f"  [Phase] Convergence Frame {frame_id}")
        
        cnt = {'pred_long': 0, 'shake_ok': 0, 'shake_fail': 0, 
               'link_success': 0, 'link_fail': 0, 'promoted': 0, 'new': 0,
               'lf_fail': 0}

        # --- Step 1: Prediction (ONLY Long Tracks) ---
        pred_objs_long = []
        long_tracks_to_shake = []
        next_long_active = []     
        cam_list = self.basic_settings._cam_list
        axis_check = self.basic_settings._axis_limit.check
        len_limit = self.LEN_LONG_TRACK
        if self.predictor is not None:
            pred_objs_long, long_tracks_to_shake = self._predict_long_tracks_nn(self.long_tracks_active, cnt)
            print(f"    [Predict NN] Ready to Shake: {len(pred_objs_long)}")
            
        else:
            batch_predicted_objs = self._batch_predict_tracks(self.long_tracks_active)
            for tr, obj in zip(self.long_tracks_active, batch_predicted_objs):
                if obj is not None:
                    c = obj._pt_center
                    if axis_check(c[0], c[1], c[2]) and obj.isReconstructable(cam_list):
                        pred_objs_long.append(obj)
                        long_tracks_to_shake.append(tr)
                        cnt['pred_long'] += 1
                        continue
                
                if len(tr._obj3d_list) >= len_limit:
                    self.exit_tracks.append(tr)
                    
            print(f"    [Predict] Ready to Shake: {len(pred_objs_long)}")

        # --- Step 2: Shake (ONLY Long Tracks) ---
        objs_for_residual = []
        if pred_objs_long:
            shaker = lpt.Shake(cam_list, self.tr_cfg)
            
            flags, shaken_objs = shaker.runShake(pred_objs_long, image_list)

            try:
                MASK_BAD = int(lpt.ObjFlag.Ghost) | int(lpt.ObjFlag.Repeated)
            except Exception:
                MASK_BAD = 0

            local_len_limit = len_limit
            local_axis_check = axis_check
            local_next_long = next_long_active
            local_inactive = self.long_tracks_inactive
            local_objs_res = objs_for_residual
            
            for flag, tr, obj_optimized in zip(flags, long_tracks_to_shake, shaken_objs):
                
                # --- 逻辑 A: 加入轨迹 (Strict: 必须是 Good) ---
                if (int(flag) & MASK_BAD) == 0:
                    tr.add_next(obj_optimized, frame_id)
                    local_next_long.append(tr)
                    cnt['shake_ok'] += 1
                else:
                    cnt['shake_fail'] += 1
                    if len(tr._obj3d_list) >= local_len_limit:
                        local_inactive.append(tr)
                
                # --- 逻辑 B: 准备擦除 (Loose: 只要位置有效就擦) ---
                if obj_optimized is not None:
                    c = obj_optimized._pt_center
                    c0, c1, c2 = c[0], c[1], c[2]
                    
                    # 彻底抛弃低效的小数组 numpy 计算，改用原生的标量 math 判断
                    if math.isfinite(c0) and math.isfinite(c1) and math.isfinite(c2):
                        if local_axis_check(c0, c1, c2):
                            local_objs_res.append(obj_optimized)

        self.long_tracks_active = deque(next_long_active)
        # --- Step 3: Residual Image Calculation ---
        image_list_residue = image_list
        
        # 计算 Long Tracks 的残差图
        if objs_for_residual:
            try:
                shaker_res = lpt.Shake(cam_list, self.tr_cfg)
                image_list_residue = shaker_res.calResidualImage(objs_for_residual, image_list)
            except Exception as e:
                print(f"    [Error] Residual Calc: {e}")

        # # --- Step 4: IPR on Residue (Using your Python function) ---
        print("    [IPR] Running on residual images (IPR)...")
        
        try:
            candidates, _ = run_ipr_on_frame(
                self.finder, self.model_S1, self.model_S2, 
                self.cpp_stb, self.tr_cfg, 
                image_list_residue, self.num_cams, phase, self.OTF
            )
        except Exception as e:
            print(f"    [Error] Python IPR failed: {e}")
            import traceback
            traceback.print_exc()
            candidates = []
             
        print(f"    [IPR] Found {len(candidates)} objects.")

        # --- Step 5: Remove Overlap (CheckRepeat) ---
        if candidates:
            candidates = self._check_repeat_py(candidates, self.long_tracks_active)

        # --- Step 6: Linking (Short Tracks -> Candidates) ---
        # 即使没有 short tracks, 也要跑流程，以便处理 new tracks
        next_short_active = deque()
        used_candidate_indices = set()
        
        n_short = len(self.short_tracks_active)
        n_cand = len(candidates)
        
        # 提取局部变量和方法，避开循环内的高频寻址开销
        local_n_init = self.n_init_frames
        local_cam_list = cam_list
        local_long_append = self.long_tracks_active.append
        local_short_append = next_short_active.append
        local_used_add = used_candidate_indices.add
        
        # 1. 尝试连接 Short Tracks
        if n_short > 0:
            if n_cand > 0:
                # 使用之前的 KDTree 严格 Linking 逻辑
                linked_indices = self._link_phase_py(self.short_tracks_active, self.long_tracks_active, candidates)
                
                for tr, cand_idx in zip(self.short_tracks_active, linked_indices):
                    if cand_idx != -1 and cand_idx not in used_candidate_indices:
                        # Link 成功
                        obj = candidates[cand_idx]
                        
                        obj.projectObject2D(local_cam_list)
                        tr.add_next(obj, frame_id)
                        local_used_add(cand_idx)
                        cnt['link_success'] += 1
                        
                        # 检查晋升
                        if len(tr._obj3d_list) >= local_n_init:
                            local_long_append(tr)
                            cnt['promoted'] += 1
                        else:
                            local_short_append(tr)
                    else:
                        # Link 失败 -> 丢弃
                        cnt['link_fail'] += 1
            else:
                 cnt['link_fail'] = n_short
        
        # 2. 创建新轨迹 (New Tracks)
        unused_indices = [i for i in range(n_cand) if i not in used_candidate_indices]
        
        for i in unused_indices:
            obj = candidates[i]
            new_tr = lpt.Track()
            obj.projectObject2D(local_cam_list)
            new_tr.add_next(obj, frame_id)
            local_short_append(new_tr)
            cnt['new'] += 1
        
        self.short_tracks_active = next_short_active
        final_long_active = deque()
        
        # --- Step 7: Pruning (Linear Fit) ---
        voxel_size = self.basic_settings._voxel_to_mm
        physical_search_radius = self.tr_cfg._pf_param.r 
        physical_epsilon = 0.2 * voxel_size 
        self.long_tracks_active, self.short_tracks_active = self._normalized_median_test_py(
            self.long_tracks_active, 
            self.short_tracks_active, 
            search_radius=physical_search_radius,
            threshold=6,                 
            epsilon=physical_epsilon        
        )

        physical_max_err = 6.0 * voxel_size
        
        for tr in self.long_tracks_active:
            pass_angle = self._check_angle_py(tr, min_cos=0.4)  # 0.8：允许最大 37 度折角
            if pass_angle and self._check_quadratic_fit_py(tr, max_err=physical_max_err):
                final_long_active.append(tr)
            # 原本的一阶拟合线性检查
            # if self._check_linear_fit_py(tr, max_err=0.5):
            #     final_long_active.append(tr)
            else:
                cnt['lf_fail'] += 1
                if len(tr._obj3d_list) >= len_limit:
                    self.long_tracks_inactive.append(tr)
                else:
                    self.exit_tracks.append(tr) # 删除
        
        self.long_tracks_active = final_long_active

        # 打印统计
        n_sa = len(self.short_tracks_active)
        n_la = len(self.long_tracks_active)
        print(f"    [Stats] Short Active: {n_sa} (LinkOk: {cnt['link_success']}, Promoted: {cnt['promoted']}, New: {cnt['new']}, Died: {cnt['link_fail']})")
        print(f"    [Stats] Long Active:  {n_la} (PredOk: {cnt['pred_long']}, ShakeOk: {cnt['shake_ok']}, LF_Fail: {cnt['lf_fail']})")
    
    def _predict_long_tracks_nn(self, tracks, cnt):
        device = next(self.predictor.parameters()).device
        pred_objs_long = []
        long_tracks_to_shake = []

        n_tracks = len(tracks)
        final_pred_positions = [None] * n_tracks

        idx_len3 = []
        pts_len3_p1 = []
        pts_len3_p2 = []

        idx_len4 = []
        pts_len4 = []

        for i, tr in enumerate(tracks):
            objs = tr._obj3d_list
            n = len(objs)
            
            if n < 3: 
                if n >= self.LEN_LONG_TRACK:
                    self.exit_tracks.append(tr)
                continue
                
            last_obj = objs[-1]
            
            if n == 3:
                idx_len3.append(i)
                pts_len3_p1.append([objs[-2]._pt_center[0], objs[-2]._pt_center[1], objs[-2]._pt_center[2]])
                pts_len3_p2.append([last_obj._pt_center[0], last_obj._pt_center[1], last_obj._pt_center[2]])
            else:
                idx_len4.append(i)
                pts_len4.append([
                    [objs[-4]._pt_center[0], objs[-4]._pt_center[1], objs[-4]._pt_center[2]],
                    [objs[-3]._pt_center[0], objs[-3]._pt_center[1], objs[-3]._pt_center[2]],
                    [objs[-2]._pt_center[0], objs[-2]._pt_center[1], objs[-2]._pt_center[2]],
                    [last_obj._pt_center[0], last_obj._pt_center[1], last_obj._pt_center[2]]
                ])

        if idx_len3:
            p1_arr = np.array(pts_len3_p1, dtype=np.float32)
            p2_arr = np.array(pts_len3_p2, dtype=np.float32)
            pred_pos_3 = p2_arr * 2.0 - p1_arr # 数学等价于 p2 + (p2 - p1)，减少一次加法运算
            
            for k, original_i in enumerate(idx_len3):
                final_pred_positions[original_i] = pred_pos_3[k]

        if idx_len4:
            pts_arr = np.array(pts_len4, dtype=np.float32)
            
            disp = np.diff(pts_arr, axis=1)
            norms = np.linalg.norm(disp, axis=2)
            S = np.max(norms, axis=1)
            S = np.maximum(S, 1e-3)
            disp_norm = disp / S[:, np.newaxis, np.newaxis]

            chunk_size = 100000 
            pred_disp_norm_list = []
            
            with torch.no_grad():
                inputs_tensor = torch.tensor(disp_norm, device=device, dtype=torch.float32)
                
                if inputs_tensor.shape[0] > chunk_size:
                    for batch_inputs in torch.split(inputs_tensor, chunk_size):
                        pred_disp_norm_list.append(self.predictor(batch_inputs).cpu().numpy())
                    pred_disp_norm = np.concatenate(pred_disp_norm_list, axis=0)
                else:
                    pred_disp_norm = self.predictor(inputs_tensor).cpu().numpy()

            last_pts = pts_arr[:, -1, :]
            pred_pos_4 = last_pts + pred_disp_norm * S[:, np.newaxis]

            for k, original_i in enumerate(idx_len4):
                final_pred_positions[original_i] = pred_pos_4[k]

        for i, tr in enumerate(tracks):
            pred_pos = final_pred_positions[i]
            if pred_pos is None:
                continue
                
            n_len = len(tr._obj3d_list)
            
            if np.isnan(pred_pos).any():
                if n_len >= self.LEN_LONG_TRACK:
                    self.exit_tracks.append(tr)
                continue
                
            px, py, pz = float(pred_pos[0]), float(pred_pos[1]), float(pred_pos[2])
            
            if self.basic_settings._axis_limit.check(px, py, pz):
                obj_new = lpt.Tracer3D()
                obj_new._pt_center = lpt.Pt3D(px, py, pz)
                
                last_obj = tr._obj3d_list[-1]
                if hasattr(last_obj, '_intensity'):
                    obj_new._intensity = last_obj._intensity
                    
                if obj_new.isReconstructable(self.basic_settings._cam_list):
                    pred_objs_long.append(obj_new)
                    long_tracks_to_shake.append(tr)
                    cnt['pred_long'] += 1
                    continue 

            if n_len >= self.LEN_LONG_TRACK:
                self.exit_tracks.append(tr)

        return pred_objs_long, long_tracks_to_shake

    def _check_linear_fit_py(self, track, max_err=5.0):
        objs = track._obj3d_list
        track_len = len(objs)
        n_init = self.n_init_frames
        n_pts = 4 if n_init > 4 else n_init
        
        if track_len < n_pts:
            return False 
            
        check_objs = objs[-n_pts:]
        pts = [o._pt_center for o in check_objs]
        act_pt = pts[-1]

        # 提前计算 X 相关的常数项
        N = float(n_pts)
        sum_x = sum(range(n_pts))
        sum_x2 = sum(i * i for i in range(n_pts))
        denom = N * sum_x2 - sum_x * sum_x
        
        if denom == 0:
            return False
            
        res_sq_sum = 0.0

        for dim in range(3):
            # 获取 Y 序列
            y_vals = [p[dim] for p in pts]
            sum_y = sum(y_vals)
            sum_xy = sum(i * y for i, y in enumerate(y_vals))
            
            # 最小二乘法求斜率和截距
            slope = (N * sum_xy - sum_x * sum_y) / denom
            intercept = (sum_y - slope * sum_x) / N
            est_val = slope * (n_pts - 1) + intercept
            
            diff = act_pt[dim] - est_val
            res_sq_sum += diff * diff
            
        return res_sq_sum <= max_err * max_err
    
    def _check_quadratic_fit_py(self, track, max_err=1.0):
        objs = track._obj3d_list
        if len(objs) < 4:
            return True
            
        c0 = objs[-4]._pt_center
        c1 = objs[-3]._pt_center
        c2 = objs[-2]._pt_center
        c3 = objs[-1]._pt_center # 真实的第4点
        
        # 公式: p3 = 3*p2 - 3*p1 + p0
        dx = c3[0] - (3.0 * c2[0] - 3.0 * c1[0] + c0[0])
        dy = c3[1] - (3.0 * c2[1] - 3.0 * c1[1] + c0[1])
        dz = c3[2] - (3.0 * c2[2] - 3.0 * c1[2] + c0[2])
        
        res_sq = dx*dx + dy*dy + dz*dz
        
        if res_sq > max_err * max_err:
            return False
            
        return True
    # 辅助函数，速度方向角检查
    def _check_angle_py(self, track, min_cos=0.9):
        """
        min_cos: 允许的最大折角对应的余弦值。
        如果是 60 度，cos(60) = 0.5。如果是 90 度，cos(90) = 0.0。
        """
        objs = track._obj3d_list
        if len(objs) < 3:
            return True
            
        c1 = objs[-3]._pt_center
        c2 = objs[-2]._pt_center
        c3 = objs[-1]._pt_center
        
        # 计算方向向量 (纯算术运算)
        v1_x = c2[0] - c1[0]
        v1_y = c2[1] - c1[1]
        v1_z = c2[2] - c1[2]
        
        v2_x = c3[0] - c2[0]
        v2_y = c3[1] - c2[1]
        v2_z = c3[2] - c2[2]
        
        dot = v1_x*v2_x + v1_y*v2_y + v1_z*v2_z
        norm1_sq = v1_x*v1_x + v1_y*v1_y + v1_z*v1_z
        norm2_sq = v2_x*v2_x + v2_y*v2_y + v2_z*v2_z
        
        if norm1_sq < 1e-10 or norm2_sq < 1e-10:
            return True
            
        if dot < 0 and min_cos >= 0:
            return False
            
        if dot * dot < min_cos * min_cos * norm1_sq * norm2_sq:
            return False
            
        return True

    def _normalized_median_test_py(self, long_tracks, short_tracks, search_radius=4.0, threshold=2.0, epsilon=0.1):
        all_tracks = list(long_tracks) + list(short_tracks)
        
        pts = []
        vels_x, vels_y, vels_z = [], [], []
        valid_indices = [] 
        
        for i, tr in enumerate(all_tracks):
            if len(tr._obj3d_list) >= 2:
                pt1_c = tr._obj3d_list[-2]._pt_center
                pt2_c = tr._obj3d_list[-1]._pt_center
                
                pts.append([pt2_c[0], pt2_c[1], pt2_c[2]])
                vels_x.append(pt2_c[0] - pt1_c[0])
                vels_y.append(pt2_c[1] - pt1_c[1])
                vels_z.append(pt2_c[2] - pt1_c[2])
                valid_indices.append(i)
                
        if len(pts) < 5:
            return long_tracks, short_tracks 
            
        pts = np.array(pts, dtype=np.float32)
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        neighbors_list = tree.query_ball_point(pts, search_radius)
        
        bad_indices = set() 
        
        def check_1d_uod(v_self, n_vals, K):
            n_vals.sort() # 原生 list 排序，对小数组极快
            mid = K // 2
            if K % 2 == 1:
                u_m = n_vals[mid]
            else:
                u_m = (n_vals[mid-1] + n_vals[mid]) * 0.5
                
            d_c = abs(v_self - u_m)
            
            d_i = [abs(v - u_m) for v in n_vals]
            d_i.sort()
            if K % 2 == 1:
                r_m = d_i[mid]
            else:
                r_m = (d_i[mid-1] + d_i[mid]) * 0.5
                
            return (d_c / (r_m + epsilon)) > threshold

        for idx, nbrs in enumerate(neighbors_list):
            K = len(nbrs) - 1 # 减去自身
            if K < 4:
                continue
                
            nx, ny, nz = [], [], []
            for n_idx in nbrs:
                if n_idx != idx:
                    nx.append(vels_x[n_idx])
                    ny.append(vels_y[n_idx])
                    nz.append(vels_z[n_idx])
                    
            if check_1d_uod(vels_x[idx], nx, K) or \
               check_1d_uod(vels_y[idx], ny, K) or \
               check_1d_uod(vels_z[idx], nz, K):
                bad_indices.add(valid_indices[idx])
                
        from collections import deque
        new_long = deque()
        for i, tr in enumerate(long_tracks):
            if i not in bad_indices:
                new_long.append(tr)
            else:
                if len(tr._obj3d_list) >= self.LEN_LONG_TRACK:
                    self.exit_tracks.append(tr) 
                    
        new_short = deque()
        offset = len(long_tracks)
        for i, tr in enumerate(short_tracks):
            if (i + offset) not in bad_indices:
                new_short.append(tr)
                
        return new_long, new_short
    def _global_spatial_pruning_py(self, long_tracks, short_tracks, search_radius=4.0, max_deviation_ratio=0.6):
        all_tracks = list(long_tracks) + list(short_tracks)
        
        pts = []
        vels = []
        valid_indices = []
        
        for i, tr in enumerate(all_tracks):
            if len(tr._obj3d_list) >= 2:
                pt1_c = tr._obj3d_list[-2]._pt_center
                pt2_c = tr._obj3d_list[-1]._pt_center
                
                p1_x, p1_y, p1_z = pt1_c[0], pt1_c[1], pt1_c[2]
                p2_x, p2_y, p2_z = pt2_c[0], pt2_c[1], pt2_c[2]
                
                pts.append([p2_x, p2_y, p2_z])
                vels.append([p2_x - p1_x, p2_y - p1_y, p2_z - p1_z])
                valid_indices.append(i)
                
        if len(pts) < 5:
            return long_tracks, short_tracks 
            
        pts = np.array(pts, dtype=np.float32)
        vels = np.array(vels, dtype=np.float32)
        
        tree = cKDTree(pts)
        neighbors_list = tree.query_ball_point(pts, search_radius)
        
        bad_indices = set() 
        
        ratio_sq = max_deviation_ratio * max_deviation_ratio
        
        for idx, nbrs in enumerate(neighbors_list):
            count = len(nbrs)
            
            if count < 4:
                continue
                
            sum_vel = np.sum(vels[nbrs], axis=0) - vels[idx]
            
            mean_vx = sum_vel[0] / (count - 1)
            mean_vy = sum_vel[1] / (count - 1)
            mean_vz = sum_vel[2] / (count - 1)
            
            v_self = vels[idx]
            diff_x = v_self[0] - mean_vx
            diff_y = v_self[1] - mean_vy
            diff_z = v_self[2] - mean_vz
            
            vel_diff_sq = diff_x*diff_x + diff_y*diff_y + diff_z*diff_z
            
            mean_vel_sq = mean_vx*mean_vx + mean_vy*mean_vy + mean_vz*mean_vz + 1e-10
            
            if vel_diff_sq > ratio_sq * mean_vel_sq:
                bad_indices.add(valid_indices[idx])
                
        new_long = deque()
        for i, tr in enumerate(long_tracks):
            if i not in bad_indices:
                new_long.append(tr)
            else:
                if len(tr._obj3d_list) >= self.LEN_LONG_TRACK:
                    self.exit_tracks.append(tr)
                    
        new_short = deque()
        offset = len(long_tracks)
        for i, tr in enumerate(short_tracks):
            if (i + offset) not in bad_indices:
                new_short.append(tr)
                
        return new_long, new_short
    
    def save_all_tracks(self, frame_id, is_initial_phase=False, clear_history=False):

        if is_initial_phase:
            folder = self.basic_settings._output_path + "InitialTrack/"
        else:
            folder = self.basic_settings._output_path + "ConvergeTrack/"
            
        if not os.path.exists(folder):
            os.makedirs(folder)
        # 2. remove_with_prefix
        # for pattern in ["LongTrackActive_*.csv", "ShortTrackActive_*.csv"]:
        #     for old_file in glob.glob(os.path.join(folder, pattern)):
        #         try:
        #             os.remove(old_file)
        #         except OSError:
        #             pass
        # 3. 执行保存
        # Active Tracks (Snapshot)
        self._save_to_csv_impl(self.long_tracks_active, folder, f"LongTrackActive_{frame_id}.csv")
        # self._save_to_csv_impl(self.short_tracks_active, folder, f"ShortTrackActive_{frame_id}.csv")
        
        # History Tracks (Buffer)
        if self.exit_tracks:
            self._save_to_csv_impl(self.exit_tracks, folder, f"ExitTrack_{frame_id}.csv")
        
        if self.long_tracks_inactive:
            self._save_to_csv_impl(self.long_tracks_inactive, folder, f"LongTrackInactive_{frame_id}.csv")
        print(f"  [Save] Tracks saved to {folder}")
        if clear_history:
            self.long_tracks_inactive.clear()
            self.exit_tracks.clear()
            print("  [Mem] Cleared Inactive/Exit track buffers.")
    def _save_to_csv_impl(self, tracks, folder, filename):
        return save_tracks(self.num_cams, tracks, folder, filename)

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a', encoding='utf-8') # 'a'表示追加模式

    def write(self, message):
        self.terminal.write(message) 
        self.log.write(message)      
        self.log.flush()             

    def flush(self):
        pass

