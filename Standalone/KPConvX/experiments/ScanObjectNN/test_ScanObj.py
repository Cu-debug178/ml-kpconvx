#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# ----------------------------------------------------------------------------------------------------------------------
#
#   Hugues THOMAS - 06/10/2023
#
#   KPConvX project: test_ScanObj.py
#       > Test script for ScanObjectNN experiments
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
import random
import torch
import numpy as np
from torch.utils.data import DataLoader

# Local libs
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)

from utils.config import load_cfg, save_cfg, get_directories
from utils.printing import frame_lines_1, underline

from models.KPConvNet import KPFCNN as KPConvFCNN
from models.KPInvNet import KPInvFCNN
from models.InvolutionNet import InvolutionFCNN
from models.KPNext import KPNeXt

from data_handlers.object_classification import ObjClassifSampler, ObjClassifCollate
from experiments.ScanObjectNN.ScanObjectNN import ScanObjectNN_cfg, ScanObjectNNDataset

from tasks.test import test_model


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Function
#       \*******************/
#


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_ScanObj_log(chosen_log,
                     new_cfg,
                     weight_path='',
                     save_visu=False,
                     test_path=''):

    ##############
    # Prepare Data
    ##############

    print('\n')
    frame_lines_1(['Data Preparation'])

    # Load dataset
    underline('Loading validation dataset')
    test_dataset = ScanObjectNNDataset(new_cfg,
                                       chosen_set='test',
                                       precompute_pyramid=True)
    
    # Calib from training data
    # test_dataset.calib_batch(new_cfg)
    # test_dataset.calib_neighbors(new_cfg)
    
    # Initialize samplers
    test_sampler = ObjClassifSampler(test_dataset)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=ObjClassifCollate,
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

    if new_cfg.model.kp_mode in ['kpconvx', 'kpconvd']:
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
    frame_lines_1(['Testing pretrained model'])

    if not test_path:
        test_path = os.path.join(chosen_log, 'test')

    # Go
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
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--max_votes', type=int)
    
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
    if args.max_votes is not None:
        if args.max_votes < 1:
            parser.error('--max_votes must be positive.')
        new_cfg.test.max_votes = args.max_votes

    # Apply the experiment seed before dataset sampling and test augmentation.
    set_seed(new_cfg.exp.seed)


    ###################
    # Define parameters
    ###################
    
    # Optionally you can change some parameters from the config file. For example:
    # new_cfg.test.batch_limit = 1
    # new_cfg.test.max_votes = 15
    # new_cfg.augment_test.anisotropic = False
    # new_cfg.augment_test.scale = [0.99, 1.01]
    # new_cfg.augment_test.flips = [0.5, 0, 0]
    # new_cfg.augment_test.rotations = 'vertical'
    # new_cfg.augment_test.color_drop = 0.0


    test_ScanObj_log(log_dir,
                     new_cfg,
                     weight_path=weights,
                     test_path=args.test_path or '')
    





