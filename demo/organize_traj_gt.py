import os
import numpy as np
import pandas as pd
import scipy.io as sio

ppp = '0.01'
pppp = '0_0100'
num_frames = 21  
dt = 0.05

root_path = rf"./test/ABC_TR_{ppp}ppp/Parameter_Particles"
all_data_blocks = []

print("========== Start loading single frame data ==========")

for t in range(num_frames):
    t_str = f"{t * dt:.3f}"
    file_name = f"Test_Particles_t_{t_str}_{pppp}ppp.mat"
    file_path = os.path.join(root_path, file_name)
    
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found")
        continue
        
    mat_data = sio.loadmat(file_path)
    
    try:
        data = mat_data['Particle_Parameter']['data'][0, 0]
    except KeyError:
        data = mat_data['Particle_Parameter'][0, 0]['data']
        
    coords = data[:, 0:3]  
    ids = data[:, 3]       
    
    frames = np.full(len(ids), t)
    
    frame_matrix = np.column_stack((ids, frames, coords))
    all_data_blocks.append(frame_matrix)
    
    print(f"Processed Frame {t} (t = {t_str})")

print("splicing and global sorting in progress...")

big_matrix = np.vstack(all_data_blocks)

df = pd.DataFrame(big_matrix, columns=['GT_ID', 'Frames', 'X', 'Y', 'Z'])
df.sort_values(by=['GT_ID', 'Frames'], inplace=True)

print("Extracting trajectory...")

grouped = df.groupby('GT_ID')
num_tracks = len(grouped)

gt_tracks_clean = np.empty(num_tracks, dtype=[
    ('Frames', 'O'), 
    ('Coords', 'O'), 
    ('GT_ID', 'O')
])

for i, (gt_id, group) in enumerate(grouped):
    gt_tracks_clean[i]['Frames'] = group['Frames'].values.reshape(-1, 1)
    gt_tracks_clean[i]['Coords'] = group[['X', 'Y', 'Z']].values
    gt_tracks_clean[i]['GT_ID']  = gt_id

save_name = os.path.join(root_path, f"GT_Tracks_{ppp}ppp.mat")

sio.savemat(save_name, {'GT_Tracks_Clean': gt_tracks_clean})

print("========== Complete ==========")
print(f"Successfully saved GT track to {save_name}")
print(f"Total of {num_tracks} real physical tracks were extracted.")