import os
import numpy as np
import cv2
from sklearn.linear_model import Ridge

input_file = r'./test/LPT_CASE3_TR_0.01/LPT_CASE03_CalibPoints.txt'
out_dir = r'./test/LPT_CASE3_TR_0.01/'
sample_img = r'./test/LPT_CASE3_TR_0.01/images/I0000_cam0.tif'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

img = cv2.imread(sample_img, cv2.IMREAD_UNCHANGED)
if img is None:
    raise ValueError(f"Unable to read image, please check path {sample_img}")
n_row, n_col = img.shape[:2]

data = np.loadtxt(input_file, skiprows=1)

X_world = data[:, 0]
Y_world = data[:, 1]  
Z_world = data[:, 2]  

radius_max = 550.0
y_min, y_max = -610, 500  
z_min, z_max = -550, 550  

poly_order = 5 
powers = []
for px in range(poly_order + 1):
    for py in range(poly_order + 1 - px):
        for pz in range(poly_order + 1 - px - py):
            powers.append((px, py, pz))

num_coeffs = len(powers)
print(f"The current polynomial is of order {poly_order}, which produces a total of {num_coeffs} coefficients.")

scale_factor = 1000.0
uv_scale = 1

ridge_solver = Ridge(alpha=1e-7, fit_intercept=False, solver='svd')

for cam_id in range(4):
    u_cam = data[:, 3 + cam_id * 2]
    v_cam = data[:, 4 + cam_id * 2]
    
    roi_mask = (X_world**2 + Z_world**2 <= radius_max**2) & \
               (Y_world >= y_min) & (Y_world <= y_max)
               
    valid_mask = (u_cam > 0) & (v_cam > 0) & ~np.isnan(u_cam) & ~np.isnan(v_cam) & roi_mask

    X_v = X_world[valid_mask]
    Y_v = Y_world[valid_mask]
    Z_v = Z_world[valid_mask]
    u_v = u_cam[valid_mask]
    v_v = v_cam[valid_mask]
    
    N = len(X_v)
    if N == 0:
        continue

    X_vn = X_v / scale_factor
    Y_vn = Y_v / scale_factor
    Z_vn = Z_v / scale_factor
    u_vn = u_v / uv_scale
    v_vn = v_v / uv_scale

    A_norm = np.zeros((N, num_coeffs), dtype=np.float64)
    for i, (px, py, pz) in enumerate(powers):
        A_norm[:, i] = (X_vn**px) * (Y_vn**py) * (Z_vn**pz)
        
    ridge_solver.fit(A_norm, u_vn)
    Cu_norm = ridge_solver.coef_
    
    ridge_solver.fit(A_norm, v_vn)
    Cv_norm = ridge_solver.coef_

    Cu_raw = np.zeros_like(Cu_norm)
    Cv_raw = np.zeros_like(Cv_norm)
    
    for i, (px, py, pz) in enumerate(powers):
        restore_multiplier = uv_scale / (scale_factor ** (px + py + pz))
        Cu_raw[i] = Cu_norm[i] * restore_multiplier
        Cv_raw[i] = Cv_norm[i] * restore_multiplier

    A_raw = np.zeros((N, num_coeffs), dtype=np.float64)
    for i, (px, py, pz) in enumerate(powers):
        A_raw[:, i] = (X_v**px) * (Y_v**py) * (Z_v**pz)
        
    u_pred = A_raw @ Cu_raw
    v_pred = A_raw @ Cv_raw
    err = np.sqrt((u_pred - u_v)**2 + (v_pred - v_v)**2)
    mean_err = np.mean(err)

    out_file = os.path.join(out_dir, f'cam{cam_id}.txt')
    with open(out_file, 'w') as f:
        f.write("# Camera Model (PINHOLE/POLYNOMIAL)\n")
        f.write("POLYNOMIAL\n")
        f.write("# Camera Calibration Error \n")
        f.write(f"{mean_err:.8f}\n")
        f.write("# Image Size (n_row,n_col)\n")
        f.write(f"{n_row},{n_col}\n")
        
        f.write("# Reference Plane (REF_X/REF_Y/REF_Z,coordinate,coordinate)\n")
        f.write(f"REF_Z,{z_min:.0f},{z_max:.0f}\n")

        f.write("# Number of Coefficients \n")
        f.write(f"{num_coeffs}\n")
        
        f.write("# U_Coeff,X_Power,Y_Power,Z_Power\n")
        for c, (px, py, pz) in zip(Cu_raw, powers):
            f.write(f"{c:.16e},{float(px):.8e},{float(py):.8e},{float(pz):.8e}\n")
            
        f.write("# V_Coeff,X_Power,Y_Power,Z_Power\n")
        for c, (px, py, pz) in zip(Cv_raw, powers):
            f.write(f"{c:.16e},{float(px):.8e},{float(py):.8e},{float(pz):.8e}\n")
            
    print(f"Cam {cam_id} Fitting complete | Number of points involved {N} | True coordinate reprojection error {mean_err:.6f} px")