#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# ----------------------------------------------------------------------------------------------------------------------
#
#   Hugues THOMAS - 06/10/2023
#
#   KPConvX project: test_S3DIS.py
#       > Test script for S3DIS experiments
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
from operator import mod
import os
import sys
import time
import signal
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

# Local libs
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)

from utils.config import load_cfg, save_cfg, get_directories
from utils.printing import frame_lines_1, underline
from utils.gpu_init import init_gpu

from models.KPConvNet import KPFCNN as KPConvFCNN
from models.KPInvNet import KPInvFCNN
from models.InvolutionNet import InvolutionFCNN
from models.KPNext import KPNeXt

from data_handlers.scene_seg import SceneSegSampler, SceneSegCollate

# from experiments.S3DIS.S3DIS import S3DIS_cfg, S3DISDataset
from experiments.S3DIS.S3DIS_rooms import S3DIR_cfg, S3DIRDataset

from tasks.test import test_model


def configure_vote_test(cfg, votes=10):
    """Apply the repository's existing full-room random-vote S3DIS protocol."""

    votes = int(votes)
    if votes < 1:
        raise ValueError('votes must be positive')

    cfg.test.batch_limit = 1
    cfg.test.in_radius = 100.0
    cfg.test.max_steps_per_epoch = 9999999
    cfg.test.max_votes = votes
    cfg.test.fixed_sampling = False

    cfg.augment_test.anisotropic = False
    cfg.augment_test.scale = [0.99, 1.01]
    cfg.augment_test.flips = [0.5, 0, 0]
    cfg.augment_test.rotations = 'vertical'
    cfg.augment_test.jitter = 0
    cfg.augment_test.color_drop = 0.0
    cfg.augment_test.chromatic_contrast = False
    cfg.augment_test.chromatic_all = False
    cfg.augment_test.pts_drop_p = -1.0
    return cfg


def profile_S3DIS(net, test_loader, cfg, on_gpu=True, get_flops=False, test_path=""):

    ############
    # Parameters
    ############

    # Choose to train on CPU or GPU
    if on_gpu and torch.cuda.is_available():
        device = init_gpu()
    else:
        device = torch.device("cpu")
        

    ####################
    # Initialize network
    ####################

    # Get the network to the device we chose
    net.to(device)
    net.eval()

    # Get number of learnable parameters of the model
    param_num = sum(param.numel() for param in net.parameters() if param.requires_grad)

    print()
    print("Model size %i" % sum(param.numel() for param in net.parameters() if param.requires_grad))
    print()


    ##################
    # Start experiment
    ##################

    with torch.no_grad():

        ############
        # Initialize
        ############

        softmax = torch.nn.Softmax(1)
        
        underline('Profiling Speed of the network')
        message =  '\n                                                          Timings        '
        message += '\n Steps |   Votes   | GPU usage |      Speed      |   In   Batch  Forw  End '
        message += '\n-------|-----------|-----------|-----------------|-------------------------'
        print(message)


        #####################
        # Network predictions
        #####################

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)
        all_speeds = []
        all_flops = []
        all_gpu_mem = []

        t1 = time.time()

        # Start validation loop
        for step, batch in enumerate(test_loader):

            # New time
            t = t[-1:]
            if 'cuda' in device.type:
                torch.cuda.synchronize(device)
            t += [time.time()]

            if 'cuda' in device.type:
                batch.to(device)

            if 'cuda' in device.type:
                torch.cuda.synchronize(device)
            t += [time.time()]

            # Forward pass
            outputs = net(batch)
            
            if 'cuda' in device.type:
                torch.cuda.synchronize(device)
            t += [time.time()]

            # Get probs and labels
            stacked_probs = softmax(outputs)
            lengths = batch.in_dict.lengths[0]
            lengths0 = batch.in_dict.lengths0
            in_inds = batch.in_dict.input_inds
            in_invs = batch.in_dict.input_invs
            cloud_inds = batch.in_dict.cloud_inds

            # Get CUDA memory stat to see what space is used on GPU
            if 'cuda' in device.type:
                cuda_stats = torch.cuda.memory_stats(device)
                used_GPU_MB = cuda_stats["allocated_bytes.all.peak"]
                _, tot_GPU_MB = torch.cuda.mem_get_info(device)
                gpu_usage = 100 * used_GPU_MB / tot_GPU_MB
                gpu_mem = used_GPU_MB
                torch.cuda.reset_peak_memory_stats(device)
            else:
                gpu_usage = 0
                gpu_mem = 0

            # # Empty GPU cache (helps avoiding OOM errors)
            # # Loses ~10% of speed but allows batch 2 x bigger.
            # torch.cuda.empty_cache()

            if 'cuda' in device.type:
                torch.cuda.synchronize(device)
            t += [time.time()]

            # Average timing
            if step < 5:
                mean_dt = np.array(t[1:]) - np.array(t[:-1])
            else:
                mean_dt = 0.8 * mean_dt + 0.2 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = ' {:5d} | {:9.2f} | {:7.1f} % | {:7.1f} stp/min | {:6.1f} {:5.1f} {:5.1f} {:5.1f}'
                print(message.format(step,
                                    test_loader.dataset.get_votes(),
                                    gpu_usage,
                                    60 / np.sum(mean_dt),
                                    1000 * mean_dt[0],
                                    1000 * mean_dt[1],
                                    1000 * mean_dt[2],
                                    1000 * mean_dt[3]))



            # Measure speed
            if 10 < step < 30:

                if get_flops:
                    N = int(torch.sum(batch.in_dict.lengths[0]))
                    B = 1
                    flops, macs, params = get_model_profile(model=net,
                                                            args=[batch],
                                                            print_profile=False,
                                                            detailed=False,         # print the detailed profile
                                                            warm_up=0,              # the number of warm-ups before measuring the time of each module
                                                            as_string=False)         # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                    print(f'Batches\tnpoints\tParams.(M)\tGFLOPs')
                    print(f'{batch_size}\t{N}\t{params / 1e6: .3f}\t{flops / (float(B) * 1e9): .2f}')
                    all_flops.append(flops)
                    if step > 15:
                        break

                else:
                    batch_size = len(batch.in_dict.lengths[0])
                    stps_per_sec = 1 / np.sum(mean_dt)
                    ins_per_sec = stps_per_sec * batch_size
                    all_speeds.append(ins_per_sec)
                    all_gpu_mem.append(gpu_mem)
                    print('throughput = {:.1f} ins/sec'.format(ins_per_sec))


            if step > 30:
                break

        t2 = time.time()


    if len(test_path) < 1:
        test_path = os.path.join(cfg.exp.log_dir, 'test')
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    if get_flops:

        # FLOPS
        avg_flops = np.mean(all_flops)  / (float(B) * 1e9)  

        report_lines = ['']
        report_lines += ['Profile']
        report_lines += ['*' * len(report_lines[1])]
        report_lines += ['']
        report_lines += ['Average FLOPs = {:.1f} G'.format(avg_flops)]
        report_lines += ['']
        report_str = frame_lines_1(report_lines)

        # Save profile
        profile_path = os.path.join(test_path, 'profile_flops.txt')
        with open(profile_path, "w") as text_file:
            text_file.write('{:^15s}'.format('FLOPs'))
            text_file.write('\n')
            text_file.write('{:^15.3f}'.format(avg_flops))
            text_file.write('\n')

    else:

        # Speed
        avg_speed = np.mean(all_speeds)
        avg_gpu_mem = np.mean(all_gpu_mem)

        report_lines = ['']
        report_lines += ['Profile']
        report_lines += ['*' * len(report_lines[1])]
        report_lines += ['']
        report_lines += ['Average throughput = {:.1f} ins/sec'.format(avg_speed)]
        report_lines += ['Average GPU Memory = {:.1f} MB'.format(avg_gpu_mem * 1e-6)]
        report_lines += ['']
        report_str = frame_lines_1(report_lines)

        # Save profile
        profile_path = os.path.join(test_path, 'profile_speed.txt')
        with open(profile_path, "w") as text_file:
            text_file.write('{:^15s} {:^15s} {:^15s}'.format('throughput', 'GPU', 'model size'))
            text_file.write('\n')
            text_file.write('{:^15.3f} {:^15.3f} {:^15d}'.format(avg_speed, avg_gpu_mem, param_num))
            text_file.write('\n')


    return


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Function
#       \*******************/
#

def test_S3DIS_log(chosen_log, new_cfg, weight_path='', save_visu=False, profile=False):

    ##############
    # Prepare Data
    ##############

    print('\n')
    frame_lines_1(['Data Preparation'])

    # Load dataset
    underline('Loading validation dataset')
    test_dataset = S3DIRDataset(new_cfg,
                                chosen_set='test',
                                precompute_pyramid=True)
    
    # Calib from training data
    # test_dataset.calib_batch(new_cfg)
    # test_dataset.calib_neighbors(new_cfg)
    
    # Initialize samplers
    test_sampler = SceneSegSampler(test_dataset)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=SceneSegCollate,
                             num_workers=new_cfg.test.num_workers,
                             pin_memory=True)


    ###############
    # Build network
    ###############

    print()
    frame_lines_1(['Model preparation'])

    underline('Loading network')

    modulated = False
    if 'mod' in new_cfg.model.kp_mode:
        modulated = True

    if new_cfg.model.kp_mode in ['kpconvx', 'kpconvd', 'kpconv', 'kpconvtest']:
        net = KPNeXt(new_cfg)

    elif new_cfg.model.kp_mode.startswith('kpconv') or new_cfg.model.kp_mode.startswith('kpmini'):
        net = KPConvFCNN(new_cfg, modulated=modulated, deformable=False)
    elif new_cfg.model.kp_mode.startswith('kpdef'):
        net = KPConvFCNN(new_cfg, modulated=modulated, deformable=True)
    elif new_cfg.model.kp_mode.startswith('kpinv'):
        net = KPInvFCNN(new_cfg)
    elif new_cfg.model.kp_mode.startswith('transformer') or new_cfg.model.kp_mode.startswith('inv_'):
        net = InvolutionFCNN(new_cfg)
    elif new_cfg.model.kp_mode.startswith('kpnext'):
        net = KPNeXt(new_cfg, modulated=modulated, deformable=False)

        
    #########################
    # Load pretrained weights
    #########################

    if weight_path:
        chosen_chkp = weight_path
    else:
        chosen_chkp = os.path.join(chosen_log, 'checkpoints', 'current_chkp.tar')


    # Load previous checkpoint
    checkpoint = torch.load(chosen_chkp, map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    print("\nModel and training state restored from:")
    print(chosen_chkp)
    print()
    
    ############
    # Start test
    ############

    print('\n')
    frame_lines_1(['Training and Validation'])

    test_path = os.path.join(chosen_log, 'test')

    # Go
    if profile:
        profile_S3DIS(net, test_loader, new_cfg, get_flops=False, test_path=test_path)
    else:
        test_model(net, test_loader, new_cfg, save_visu=save_visu, test_path=test_path)

    return


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#


if __name__ == '__main__':

    ###############
    # Get arguments
    ###############

    # This script accepts the following arguments:
    #   --log_path: Path to the log folder that we want to test
    #   --weight_path: Path to the weight file that we want to test
    #
    # If you provide the weight path, it has to be in the log_path folder. 
    # It allows you to choose a specific weight file from the log folder.

    # Add argument here to handle it
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str)
    parser.add_argument('--weight_path', type=str)
    parser.add_argument('--dataset_path', type=str)    
    parser.add_argument('--profile', action='store_true')
    
    # Read arguments
    args = parser.parse_args()
    log_dir = args.log_path
    weights = args.weight_path
    
    assert os.path.exists(log_dir), f"Log folder {log_dir} does not exist."
    if weights:
        assert os.path.exists(weights), f"Weight file {weights} does not exist."
        assert log_dir in weights, f"Weight file {weights} is not in log folder {log_dir}."
    

    #############
    # Load config
    #############

    # Configuration parameters
    new_cfg = load_cfg(log_dir)

    # Change dataset path if provided
    if args.dataset_path:
        new_cfg.data.path = args.dataset_path


    ###################
    # Define parameters
    ###################
    
    # Optionally you can change some parameters from the config file. For example:
        
    configure_vote_test(new_cfg, votes=10)


    test_S3DIS_log(log_dir, new_cfg, weight_path=weights, profile=args.profile)
    





