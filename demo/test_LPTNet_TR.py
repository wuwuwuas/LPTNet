import os
import sys
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import torch
import argparse
from collections import deque
import pyopenlpt as lpt 
from sub_modules import tracker_STB, Logger, build_tracker_NN
from STB_python import PythonSTB

if __name__ == "__main__":
    ppp = '0.05'
    config_file = f'./test/ISO_TR_{ppp}ppp/config_python_{ppp}_NND_GOTrack+NNP2.txt'
    log_file = f'./test/ISO_TR_{ppp}ppp/log_NND_GOTrack+NNP2.txt'

    NND = True
    NNT = True
    NNP = True
    OTF = False
    n_segments = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ',device)
    use_multi_gpu = torch.cuda.device_count() > 1

    if NND:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        if project_root not in sys.path:
            sys.path.append(project_root)
        NN_detec_path = os.path.join(project_root, 'NN_detec')
        if NN_detec_path not in sys.path:
            sys.path.append(NN_detec_path)
        from NN_detec.Detector_S1 import ParticleDetectionNet_Passthrough
        from NN_detec.Detector_S2 import PatchRegressor_Slots
        model_path_S1 = r"NN_detec/checkpoints/s1/S1_checkpoint.pth"
        model_path_S2 = r"NN_detec/checkpoints/s2/S2_checkpoint.pth"
        
        max_particles = 7
        patch_size = 7
        batch_size = 2000
        model_S1 = ParticleDetectionNet_Passthrough(in_channels=1, num_p=1)
        model_S2 = PatchRegressor_Slots(in_ch=1, patch_size=7, d_model=64, K=5)
        if os.path.exists(model_path_S1) and os.path.exists(model_path_S2):
            checkpoint_S1 = torch.load(model_path_S1, map_location=device)
            checkpoint_S2 = torch.load(model_path_S2, map_location=device)
            model_S1.load_state_dict(checkpoint_S1)
            model_S2.load_state_dict(checkpoint_S2)
            model_S1.to(device)
            model_S2.to(device)
            if use_multi_gpu:
                model_S1 = torch.nn.DataParallel(model_S1)
                model_S2 = torch.nn.DataParallel(model_S2)
            model_S1.eval()
            model_S2.eval()
        else:
            print("Error: Model checkpoints not found.")
            sys.exit(1)
    else:
        model_S1 = None
        model_S2 = None

    if NNT:
        parser = argparse.ArgumentParser()
        parser.add_argument("--weights_GNN", type=str, 
                            default="./GOTrack3D/weights_GotFlow3D/checkpoints/best_checkpoint.params", help="Path to saved checkpoint.")
        parser.add_argument('--delta_t', type=int, default=1, help='calculation interval for particle tracking')
        parser.add_argument('--max_points', help='maximum number of points sampled from a point cloud', default=10000,type=int)
        parser.add_argument('--corr_levels', help='number of correlation pyramid levels', default=3, type=int)
        parser.add_argument('--base_scales', help='voxelize base scale', default=0.25,  type=float)
        parser.add_argument('--truncate_k', help='value of truncate_k in corr block', default=2000, type=int)
        parser.add_argument('--iters', help='number of iterations in GRU module', default=3, type=int)
        parser.add_argument('--nb_iter', type=int, default=10, help='Number of unrolled iterations in the Sinkhorn algorithm')
        parser.add_argument('--tracking_mode', type=str, default='GOTrack+', 
                            choices=['GOTrack+', 'GOTrack'], help='output directory for results')
        parser.add_argument('--candidates', type=int, default=7, help='Number of candidate particles')
        parser.add_argument('--neighbor_similarity', type=int, default=8, help='Number of neighbor particles for similarity checking')
        parser.add_argument('--threshold_similarity', type=int, default=4, help='Threshold for similarity checking')  
        parser.add_argument('--neighbor_outlier', type=int, default=9, help='Number of neighbor particles for outlier removal')
        parser.add_argument('--threshold_outlier', type=int, default=2, help='Threshold for outlier removal')  
        
        args = parser.parse_args()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        if project_root not in sys.path:
            sys.path.append(project_root)
        gotrack_path = os.path.join(project_root, 'GOTrack3D')
        if gotrack_path not in sys.path:
            sys.path.append(gotrack_path)
        from GOTrack3D.model.ParticleMatch import Tracking

        file = torch.load(args.weights_GNN, map_location=device)
        tracker = Tracking(args).to(device)
        tracker.load_state_dict(file["state_dict"])
        if use_multi_gpu:
            tracker = torch.nn.DataParallel(tracker)
        tracker.eval()
        nn_tracker_func = build_tracker_NN(tracker, nb_iter=args.nb_iter)
        Track = nn_tracker_func
    else:
        Track = tracker_STB

    sys.stdout = Logger(log_file, sys.stdout)
    finder = lpt.ObjectFinder2D()

    if NNP:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        if project_root not in sys.path:
            sys.path.append(project_root)
        NN_predictor_path = os.path.join(project_root, 'NN_predictor')
        if NN_predictor_path not in sys.path:
            sys.path.append(NN_predictor_path)
        from NN_pred.NN_predictor import LagrangeGRUPredictor
        predictor = LagrangeGRUPredictor(input_dim=3, hidden_dim=64, num_layers=2)
        weight_path = r"NN_pred/checkpoint/lagrange_gru.pth"
        predictor.load_state_dict(torch.load(weight_path, map_location=device))
        predictor.to(device)
        if use_multi_gpu:
            predictor = torch.nn.DataParallel(predictor)
        predictor.eval()
    else:
        predictor = None

    stb_runner = PythonSTB(config_file, finder, model_S1, model_S2, Track, predictor, n_segments, OTF)
    stb_runner.run()
