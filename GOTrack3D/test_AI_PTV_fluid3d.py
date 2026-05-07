import numpy as np
import torch
import glob
import matplotlib.pyplot as plt
import os
import argparse
import logging
from model.ParticleMatch import Tracking
from scipy.io import savemat
import scipy.io as scio
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from tools.metric import compute_epe2

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
    parser.add_argument('--name', type=str, default='jet',
                        help='name of experiment')
    parser.add_argument('--result_plot', type=int, default=1,
                        choices=[0, 1],
                        help='Whether to plot the results. 0: No; 1: Yes')

    parser.add_argument('--max_points', help='maximum number of points sampled from a point cloud',
                        default=10000,type=int)
    parser.add_argument('--corr_levels', help='number of correlation pyramid levels', default=3, type=int)
    parser.add_argument('--base_scales', help='voxelize base scale', default=6,  type=float)
    parser.add_argument('--truncate_k', help='value of truncate_k in corr block', default=2000, type=int)
    parser.add_argument('--iters', help='number of iterations in GRU module', default=12, type=int)
    parser.add_argument('--nb_iter', type=int, default=30,
                        help='Number of unrolled iterations in the Sinkhorn algorithm')

    parser.add_argument('--tracking_mode', type=str, default='GOTrack+',
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
    parser.add_argument('--threshold_outlier', type=int, default=2,
                        help='Threshold for outlier removal')  
    
    parser.add_argument('--dve', type=int, default=1,
                        help='1: use') 
    parser.add_argument('--linear_corr_conf', type=int, default=1,
                        help='1: use') 
    
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


    path2data = 'D:\OpenLPT_GUI\GOTrack3D\data'
    from deepptv.fluidflow import FluidFlowDataset,FluidFlowDataset_GOTversion
    from deepptv.generic import Batch
    dataset = FluidFlowDataset_GOTversion(root_dir=path2data, nb_points=2048, all_points=1, mode='test', nb_examples=-1)
        
    testloader = DataLoader(
        dataset,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        num_workers=0,
        collate_fn=Batch,
        drop_last=False,
    )
    
    num_batch = len(testloader)

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
    epe_test = []
    outlier_test = []
    acc3dRelax_test = []
    acc3dStrict_test = []

    for i, batch_data in enumerate(tqdm(testloader)):    
        batch_data = batch_data.to(device, non_blocking=True)
        with torch.no_grad():
            tracker.eval()
            est_flow, flow_gri, flow_mask = tracker(batch_data['sequence'], args.nb_iter)
        
        trajectories = (est_flow != 0).any(dim=2).sum(dim=1)
        print('Number of trajectories tracked:  ', trajectories.item())
        
        
        # if args.dve == 1:
        #     from models.refiner import Refiner
        #     from tools import reconstruction as R
        #     from knn_cuda import KNN
        #     knn_candidate = KNN(k=12+1, transpose_mode=True)
        #     _, sub_candidate = knn_candidate(batch_data['sequence'][1], batch_data['sequence'][0])
            
        #     refiner = Refiner(est_flow.shape, est_flow.device)
        #     test_time_optimizer = torch.optim.Adam(refiner.parameters(), lr=1e-3)
            
        #     source_cross_nn_weight, _, source_cross_nn_idx, _, _, _ = \
        #         R.get_s_t_neighbors(12, transport, sim_normalization="none", s_only=True, nn_idx=sub_candidate)
                
                
        #     # Target point cloud cross reconstruction
        #     cross_weight_sum = source_cross_nn_weight.sum(-1, keepdim=True)
        #     source_cross_nn_weight_normalized = source_cross_nn_weight / (cross_weight_sum + 1e-8)
        #     target_cross_recon = R.reconstruct(batch_data['sequence'][1], source_cross_nn_idx, source_cross_nn_weight_normalized, 12)    
            
        #     # cross_nn_sim, _, _, _ = R.get_s_t_topk(transport, 12, s_only=True, nn_idx=source_cross_nn_idx)
            
        #     nn_sim_weighted = cross_nn_sim * source_cross_nn_weight_normalized
        #     nn_sim_weighted = torch.sum(nn_sim_weighted, dim=2)
        #     if args.linear_corr_conf == 1:
        #         corr_conf = (nn_sim_weighted + 1) / 2
        #     else:
        #         corr_conf = torch.clamp_min(nn_sim_weighted, 0.0)
            
        #     recon_flow = target_cross_recon - batch_data['sequence'][0]
            
        #     refined_flow, refine_metrics_curr, duration_curr = refiner.refine_flow(batch_data, est_flow.detach(), corr_conf.detach(), test_time_optimizer, args)
        #     est_flow = refined_flow.detach()
        
        
        
        i_str = str(i).zfill(4)
        # savemat(f'{save_path}/result_{i_str}.mat', data_dict)
        epe, acc3d_strict, acc3d_relax, outlier = compute_epe2(flow_gri, batch_data)
        epe_test.append(epe)
        outlier_test.append(outlier)
        acc3dRelax_test.append(acc3d_relax)
        acc3dStrict_test.append(acc3d_strict)
        savemat('pre/output'+str(i)+'.mat', {
        'flow_gri': flow_gri.cpu().numpy(),
        'pos': batch_data['sequence'][0].cpu().numpy(),
        'gt_flow': batch_data["ground_truth"][1].cpu().numpy()
        })
    print('Test Result: EPE: {:.5f} Outlier: {:.5f} Acc3dRelax: {:.5f} Acc3dStrict: {:.5f}'.format(
        np.array(epe_test).mean(),
        np.array(outlier_test).mean(),
        np.array(acc3dRelax_test).mean(),
        np.array(acc3dStrict_test).mean()
    ))   
                
if __name__ == '__main__':
    main()







