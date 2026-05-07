import numpy as np
import torch
import glob
# import matplotlib.pyplot as plt
import os
import argparse
import logging
from model.ParticleMatch import Tracking
from scipy.io import savemat
import scipy.io as scio
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default=0, type=int,
                        help='index of gpu used')

    parser.add_argument("--weights_GNN", type=str, 
                        default="D:\OpenLPT_GUI\GOTrack3D\weights_GotFlow3D\checkpoints\\best_checkpoint.params",  
                        help="Path to saved checkpoint.")
    
    parser.add_argument('--delta_t', type=int, default=1,
                        help='calculation interval for particle tracking')

    parser.add_argument('--output_dir_ckpt', type=str, default='./result/',
                        help='output directory for results')
    parser.add_argument('--data_path', type=str, default='./data/PIV4E_01/',
                        help='dataset for test')
    parser.add_argument('--img_type', type=str, default='tif',
                        help='image format')
    parser.add_argument('--name', type=str, default='cylinder',
                        help='name of experiment')
    parser.add_argument('--result_plot', type=int, default=1,
                        choices=[0, 1],
                        help='Whether to plot the results. 0: No; 1: Yes')

    parser.add_argument('--max_points', help='maximum number of points sampled from a point cloud',
                        default=10000,type=int)
    parser.add_argument('--corr_levels', help='number of correlation pyramid levels', default=3, type=int)
    parser.add_argument('--base_scales', help='voxelize base scale', default=0.25,  type=float)
    parser.add_argument('--truncate_k', help='value of truncate_k in corr block', default=2000, type=int)
    parser.add_argument('--iters', help='number of iterations in GRU module', default=12, type=int)
    parser.add_argument('--nb_iter', type=int, default=30,
                        help='Number of unrolled iterations in the Sinkhorn algorithm')

    parser.add_argument('--tracking_mode', type=str, default='GOTrack',
                        choices=['GOTrack+', 'GOTrack'],
                        help='output directory for results')
    parser.add_argument('--candidates', type=int, default=7,
                        help='Number of candidate particles')
    
    parser.add_argument('--neighbor_similarity', type=int, default=8,
                        help='Number of neighbor particles for similarity checking')
    parser.add_argument('--threshold_similarity', type=int, default=4,
                        help='Threshold for similarity checking')  
    
    parser.add_argument('--neighbor_outlier', type=int, default=9,
                        help='Number of neighbor particles for outlier removal')
    parser.add_argument('--threshold_outlier', type=int, default=1,
                        help='Threshold for outlier removal')  
    
    args = parser.parse_args()
    print('args parsed')

    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    evaluate(args)


def evaluate(args):
    device = 'cuda:' + str(args.gpu)

    def read_xyz_from_txt(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()[2:20000]
        return np.array([list(map(float, line.strip().split()[:3])) for line in lines]), np.array([list(map(float, line.strip().split()[6:7])) for line in lines])


    pos0,id0 = read_xyz_from_txt('D:\GOTrack3D\\test_data\cylinder/LPT_position_t_001.txt')
    pos1,id1 = read_xyz_from_txt('D:\GOTrack3D\\test_data\cylinder/LPT_position_t_002.txt')
    pos2,id2 = read_xyz_from_txt('D:\GOTrack3D\\test_data\cylinder/LPT_position_t_003.txt')
    
    test_pc0 = torch.from_numpy(pos0).to(torch.float32).unsqueeze(0)
    test_pc1 = torch.from_numpy(pos1).to(torch.float32).unsqueeze(0)
    test_pc2 = torch.from_numpy(pos2).to(torch.float32).unsqueeze(0)
    
    test_data = TensorDataset(test_pc0[:,0:2000,:],test_pc1[:,0:1998,:])
    print('total number of test samples:', test_pc0.shape)
    test_data = DataLoader(test_data, batch_size=1, shuffle=False)


    save_path = args.output_dir_ckpt + args.name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file = torch.load(args.weights_GNN,map_location={'cuda:3': 'cuda:0','cuda:2': 'cuda:0','cuda:1': 'cuda:0'})
    
    tracker = Tracking(args).to(device)
    weight_path = args.weights_GNN
    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path)
        tracker.load_state_dict(file["state_dict"])
        print('Load checkpoint from {}'.format(weight_path))
        print('Checkpoint epoch {}'.format(checkpoint['epoch']))
        logging.info('Load checkpoint from {}'.format(weight_path))
    else:
        raise RuntimeError(f"=> No checkpoint found at '{weight_path}")
    # Validation
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_data)):    
            pos1 = batch[0].to(device)
            pos2 = batch[1].to(device)
            # tracking
            print('------------ Particle Tracking--------------')
            # size = (pos1[:, :,0].max()-pos1[:, :,0].min())*(pos1[:, :,1].max()-pos1[:, :,1].min())*(pos1[:, :,2].max()-pos1[:, :,2].min())
            # scale = torch.sqrt((pos1.shape[0]/size)/(500/1800))
            min_val = pos1.amin(dim=1, keepdim=True)  # [1,1,3]
            max_val = pos1.amax(dim=1, keepdim=True)  # [1,1,3]
            
            scale = max_val - min_val
            scale[scale == 0] = 1.0
            normalized1 = (pos1 - min_val) / scale
            normalized2 = (pos2 - min_val) / scale
            pc1 = normalized1 * 6.28 - 3.14
            pc2 = normalized2 * 6.28 - 3.14

            batch_data = {"sequence": [pc1, pc2]}   
            for key in batch_data.keys():
                batch_data[key] = [d.to(device) for d in batch_data[key]]
            with torch.no_grad():
                tracker.eval()
                est_flow, flow_gri, flow_mask = tracker(batch_data['sequence'], args.nb_iter)
            
            trajectories = (est_flow != 0).any(dim=2).sum(dim=1)
            print('Number of trajectories tracked:  ', trajectories.item())
            
            est_flow = est_flow / 6.28 * scale
            flow_gri = flow_gri / 6.28 * scale
            pc1 = (pc1 + 3.14)/6.28 * scale + min_val
            pc2 = (pc2 + 3.14)/6.28 * scale + min_val
            
            pc1 = pc1.cpu().numpy()
            pc2 = pc2.cpu().numpy()
            est_flow = est_flow.cpu().numpy()
            
            # data_dict = {'pc1': pc1, 'pc2': pc2, 'est_flow': est_flow, 'flow_gri':flow_gri.cpu().numpy(),'scale':scale.cpu().numpy()}
            # i_str = str(i).zfill(4)
            # savemat(f'{save_path}/result_{i_str}.mat', data_dict)
                
if __name__ == '__main__':
    main()







