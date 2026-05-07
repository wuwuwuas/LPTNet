import os
import cv2
import numpy as np
import scipy.linalg
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def reproj_error_poly(params, X3D, pts2D):
    rvec = params[0:3]
    tvec = params[3:6]
    fx, fy, cx, cy = params[6:10]
    
    dist_coeffs = np.array([params[10], params[11], params[12], params[13], 0.0], dtype=np.float64)
    
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    proj_pts, _ = cv2.projectPoints(np.ascontiguousarray(X3D), rvec, tvec, K, dist_coeffs)
    proj_pts = proj_pts.reshape(-1, 2)
    return (proj_pts - pts2D).ravel()


def plot_error_distribution(cam_idx, pts2D, dx, dy, errors, n_row, n_col, removed_count):
    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(f'Camera {cam_idx} Error Analysis (Cleaned, Removed {removed_count} outliers)', fontsize=16, fontweight='bold')
    
    ax1 = fig.add_subplot(131)
    ax1.scatter(dx, dy, alpha=0.6, edgecolors='w', linewidths=0.5)
    ax1.axhline(0, color='r', linestyle='--', linewidth=1)
    ax1.axvline(0, color='r', linestyle='--', linewidth=1)
    ax1.set_title('Error Scatter (dx vs dy)')
    ax1.set_xlabel('dx (pixels)')
    ax1.set_ylabel('dy (pixels)')
    ax1.axis('equal')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    ax2 = fig.add_subplot(132)
    ax2.hist(errors, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(errors), color='r', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(errors):.3f}')
    ax2.set_title('Error Magnitude Histogram')
    ax2.set_xlabel('Error Magnitude (pixels)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    ax3 = fig.add_subplot(133)
    q = ax3.quiver(pts2D[:, 0], pts2D[:, 1], dx, dy, errors, 
                   cmap='jet', angles='xy', scale_units='xy')
    fig.colorbar(q, ax=ax3, label='Error Magnitude (px)')
    ax3.set_xlim([0, n_col])
    ax3.set_ylim([n_row, 0]) 
    ax3.set_title('Spatial Error Distribution')
    ax3.set_xlabel('Image X (pixels)')
    ax3.set_ylabel('Image Y (pixels)')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

def run_ultimate_calibration():
    # # ISO
    # input_file = r'./test/ISO_TR_0.01ppp/Parameter_CalibPoints/Test_CalibPoints_t_0.000_0_0100ppp.txt'
    # out_dir = r'./test/ISO_TR_0.01ppp/'
    # sample_img = r'./test/ISO_TR_0.01ppp/Img_Cam0/Test_Img_t_0.000_0_0100ppp_cam0.tif'

    # ABC
    input_file = r'test/Density/0.010/Parameter_CalibPoints/Test_CalibPoints_Sample_00000_0_010ppp.txt'
    out_dir = r'test/Density/'
    sample_img = r'test/Density/0.010/Img/Test_Img_Sample_00000_0_010ppp_cam1.tif'
    OUTLIER_THRESHOLD = 2.5 

    if not os.path.exists(out_dir): os.makedirs(out_dir)

    img = cv2.imread(sample_img, cv2.IMREAD_UNCHANGED)
    n_row, n_col = img.shape[:2]

    data = np.loadtxt(input_file, skiprows=1)
    X_world_all = data[:, 0:3]
    num_pts_total = X_world_all.shape[0]

    final_cam_params = []

    for cam_idx in range(4):
        col_start = 3 + cam_idx * 2
        x_cam_all = data[:, col_start:col_start+2]

        mean_X = np.mean(X_world_all, axis=0)
        scale_X = np.sqrt(3) / np.mean(np.linalg.norm(X_world_all - mean_X, axis=1))
        T_world = np.array([[scale_X,0,0,-scale_X*mean_X[0]], [0,scale_X,0,-scale_X*mean_X[1]], [0,0,scale_X,-scale_X*mean_X[2]], [0,0,0,1]])
        X_n = (T_world @ np.hstack((X_world_all, np.ones((num_pts_total, 1)))).T).T

        mean_x = np.mean(x_cam_all, axis=0)
        scale_x = np.sqrt(2) / np.mean(np.linalg.norm(x_cam_all - mean_x, axis=1))
        T_img = np.array([[scale_x,0,-scale_x*mean_x[0]], [0,scale_x,-scale_x*mean_x[1]], [0,0,1]])
        x_n = (T_img @ np.hstack((x_cam_all, np.ones((num_pts_total, 1)))).T).T

        A = np.zeros((2 * num_pts_total, 12))
        for i in range(num_pts_total):
            X, Y, Z, _ = X_n[i]; u, v, _ = x_n[i]
            A[2*i]   = [X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u]
            A[2*i+1] = [0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v]

        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        P = np.linalg.inv(T_img) @ Vt[-1].reshape(3, 4) @ T_world
        
        K_init, R_init = scipy.linalg.rq(P[:, 0:3])
        T_sign = np.diag(np.sign(np.diag(K_init)))
        K_init = K_init @ T_sign; R_init = T_sign @ R_init
        if np.linalg.det(R_init) < 0: R_init = -R_init; K_init = -K_init
        P = P / K_init[2, 2]; K_init = K_init / K_init[2, 2]
        t_init = np.linalg.inv(K_init) @ P[:, 3]
        rot_vec_init, _ = cv2.Rodrigues(R_init)

        initial_params = np.hstack([
            rot_vec_init.flatten(), t_init, 
            [K_init[0,0], K_init[1,1], K_init[0,2], K_init[1,2]], 
            [0.0, 0.0, 0.0, 0.0]  
        ])

        res_pass1 = least_squares(reproj_error_poly, initial_params, method='lm', args=(X_world_all, x_cam_all))
        residuals1 = reproj_error_poly(res_pass1.x, X_world_all, x_cam_all).reshape(-1, 2)
        errors1 = np.linalg.norm(residuals1, axis=1)

        inlier_mask = errors1 < OUTLIER_THRESHOLD
        X_world_clean = X_world_all[inlier_mask]
        x_cam_clean = x_cam_all[inlier_mask]
        removed_count = num_pts_total - np.sum(inlier_mask)

        res_pass2 = least_squares(reproj_error_poly, res_pass1.x, method='lm', args=(X_world_clean, x_cam_clean))
        opt_params = res_pass2.x

        rot_vec = opt_params[0:3]
        t = opt_params[3:6]
        fx, fy, cx, cy = opt_params[6:10]
        dist_coeffs = np.array([opt_params[10], opt_params[11], opt_params[12], opt_params[13], 0.0])
        
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        R, _ = cv2.Rodrigues(rot_vec)
        inv_R = np.linalg.inv(R); inv_t = -inv_R @ t
        
        final_cam_params.append({'K': K, 'dist': dist_coeffs, 'rvec': rot_vec, 'tvec': t})

        residuals_final = reproj_error_poly(opt_params, X_world_clean, x_cam_clean).reshape(-1, 2)
        dx, dy = residuals_final[:, 0], residuals_final[:, 1]
        errors_final = np.linalg.norm(residuals_final, axis=1)
        err_mean = np.mean(errors_final)
        
        print(f"【Cam {cam_idx}】 Average {err_mean:.6f} px | Removed {removed_count} items")

        plot_error_distribution(cam_idx, x_cam_clean, dx, dy, errors_final, n_row, n_col, removed_count)

        out_file = os.path.join(out_dir, f'cam{cam_idx}.txt')
        with open(out_file, 'w') as f:
            f.write("# Camera Model: (PINHOLE/POLYNOMIAL)\nPINHOLE\n")
            f.write(f"# Camera Calibration Error: \n{err_mean:.6f}\n")
            f.write("# Pose Calibration Error: \nNone\n")
            f.write(f"# Image Size: (n_row,n_col)\n{n_row},{n_col}\n")
            f.write("# Camera Matrix: \n")
            f.write(f"{K[0,0]:.12f},{K[0,1]:.12f},{K[0,2]:.12f}\n")
            f.write(f"{K[1,0]:.12f},{K[1,1]:.12f},{K[1,2]:.12f}\n")
            f.write(f"{K[2,0]:.12f},{K[2,1]:.12f},{K[2,2]:.12f}\n")
            f.write("# Distortion Coefficients: \n")
            f.write(f"{dist_coeffs[0]:.12f},{dist_coeffs[1]:.12f},{dist_coeffs[2]:.12f},{dist_coeffs[3]:.12f},{dist_coeffs[4]:.12f}\n")
            f.write("# Rotation Vector: \n")
            f.write(f"{rot_vec[0]:.12f},{rot_vec[1]:.12f},{rot_vec[2]:.12f}\n")
            f.write("# Rotation Matrix: \n")
            f.write(f"{R[0,0]:.6f},{R[0,1]:.6f},{R[0,2]:.6f}\n")
            f.write(f"{R[1,0]:.6f},{R[1,1]:.6f},{R[1,2]:.6f}\n")
            f.write(f"{R[2,0]:.6f},{R[2,1]:.6f},{R[2,2]:.6f}\n")
            f.write("# Inverse of Rotation Matrix: \n")
            f.write(f"{inv_R[0,0]:.12f},{inv_R[0,1]:.12f},{inv_R[0,2]:.12f}\n")
            f.write(f"{inv_R[1,0]:.12f},{inv_R[1,1]:.12f},{inv_R[1,2]:.12f}\n")
            f.write(f"{inv_R[2,0]:.12f},{inv_R[2,1]:.12f},{inv_R[2,2]:.12f}\n")
            f.write("# Translation Vector: \n")
            f.write(f"{t[0]:.6f},{t[1]:.6f},{t[2]:.6f}\n")
            f.write("# Inverse of Translation Vector: \n")
            f.write(f"{inv_t[0]:.12f},{inv_t[1]:.12f},{inv_t[2]:.12f}\n")

if __name__ == "__main__":
    run_ultimate_calibration()