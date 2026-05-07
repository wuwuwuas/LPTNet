import os
import pandas as pd
import numpy as np
import scipy.io as sio

ppp = '0.01'
last_frame = 20
root_path = rf"./test/ABC_TR_{ppp}ppp/results {ppp}ppp_NND_NNP"
converge_path = os.path.join(root_path, "ConvergeTrack")

# ppp = '0.04'
# last_frame = 49
# root_path = rf"../test/LPT_CASE3_TR_{ppp}/results NND"
# converge_path = os.path.join(root_path, "ConvergeTrack")

base_names = [
    "LongTrackActive",
    "LongTrackInactive",
    "ExitTrack",
    "ShortTrackActive",
]

all_frames = []
all_coords = []
all_source = []
all_orig_id = []

print("========== Start extracting predicted trajectory ==========")
for base in base_names:
    csv_name = f"{base}_{last_frame}.csv"
    parquet_name = f"{base}_{last_frame}.parquet"
    
    csv_path = os.path.join(converge_path, csv_name)
    parquet_path = os.path.join(converge_path, parquet_name)
    
    df = pd.DataFrame()
    used_filename = ""
    
    if os.path.exists(parquet_path):
        print(f"Reading and processing {parquet_name} ...")
        df = pd.read_parquet(parquet_path)
        used_filename = parquet_name
    elif os.path.exists(csv_path):
        print(f"Reading and processing {csv_name} ...")
        df = pd.read_csv(csv_path)
        used_filename = csv_name
    else:
        print(f"No CSV or Parquet file for {base}_{last_frame} found, skipping...")
        continue
        
    if df.empty:
        continue
        
    df.sort_values(by=['TrackID', 'FrameID'], inplace=True)
    
    grouped = df.groupby('TrackID')
    
    for track_id, group in grouped:
        all_frames.append(group['FrameID'].values)
        all_coords.append(group[['WorldX', 'WorldY', 'WorldZ']].values)
        all_source.append(used_filename)
        all_orig_id.append(track_id)

if len(all_frames) == 0:
    print("No trajectory data was extracted.")
else:
    pred_tracks = np.empty(len(all_frames), dtype=[
        ('Frames', 'O'), 
        ('Coords', 'O'), 
        ('Source', 'O'), 
        ('Original_TrackID', 'O')
    ])

    for i in range(len(all_frames)):
        pred_tracks[i]['Frames'] = all_frames[i].reshape(-1, 1)
        pred_tracks[i]['Coords'] = all_coords[i]
        pred_tracks[i]['Source'] = all_source[i]
        pred_tracks[i]['Original_TrackID'] = all_orig_id[i]

    save_name = os.path.join(root_path, f"Tracks_{ppp}ppp.mat")
    sio.savemat(save_name, {'Pred_Tracks': pred_tracks})

    print(f"The predicted trajectory was successfully saved to {save_name}")
    print(f"Total of {len(all_frames)} predicted trajectories were extracted (including true matches and ghosts).")