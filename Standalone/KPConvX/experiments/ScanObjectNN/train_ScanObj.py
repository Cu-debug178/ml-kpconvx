#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# ----------------------------------------------------------------------------------------------------------------------
#
#   Hugues THOMAS - 06/10/2023
#
#   KPConvX project: train_ScanObj.py
#       > Training script for ScanObjectNN experiments
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
from decimal import MAX_PREC
from operator import mod
import os
import sys
import time
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

# Fixing RuntimeError: received 0 items of ancdata
# See https://github.com/Project-MONAI/MONAI/issues/701
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


# Local libs
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)

from utils.config import init_cfg, load_cfg, save_cfg, get_directories
from utils.printing import frame_lines_1, underline
from utils.gpu_init import init_gpu

from models.KPNext import KPNeXt
from models.KPConvNet import KPFCNN as KPConvFCNN
from models.KPInvNet import KPInvFCNN
from models.InvolutionNet import InvolutionFCNN

from data_handlers.object_classification import ObjClassifSampler, ObjClassifCollate

from experiments.ScanObjectNN.ScanObjectNN import ScanObjectNN_cfg, ScanObjectNNDataset

from tasks.trainval import train_and_validate


# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#


def my_config():
    """
    Override the parameters you want to modify for this dataset
    """

    cfg = init_cfg()
    cfg.exp.seed = 57106803  # Match the aligned Pointcept/S3DIS baseline run.

    # Network parameters
    # ------------------

    # OLD
    # cfg.model.layer_blocks = (4,  6,  9, 12, 4)   # KPNetX-S
    # cfg.model.layer_blocks = (3,  3,  9, 12, 3)   # KPNetX-S
    # cfg.model.layer_blocks = (4,  4, 12, 20,  4)    # KPNetX-L
    # cfg.model.layer_blocks = (6,  9,  9,  4)    # strange test
    # cfg.model.layer_blocks = (3, 9, 12, 3, 3)   # KPNetX-S bis
    # cfg.model.layer_blocks = (3, 3, 9, 3, 3)   # KPNetX-S bisbis

    cfg.model.layer_blocks = (3,  3,  9, 12, 3)   # L
    # cfg.model.layer_blocks = (2,  2,  2,  8, 2)   # S


    cfg.model.norm = 'batch' # batch, layer
    cfg.model.init_channels = 64  # 48, 64, 80, 96
    cfg.model.channel_scaling = 1.41  # 1.41 (1/2) or 1.59 (2/3)

    cfg.model.kp_mode = 'kpconvd'       # Choose ['kpconv', 'kpdef', 'kpinv', 'kpinvx'].
                                        # Choose ['inv_v1', 'inv_v2', 'inv_v3', 'inv_v4', 'transformer']
                                        # Choose ['kpconv-mod', 'kpdef-mod', 'kpconv-geom'] for modulations
                                        # Choose ['kpconv-depth'] for depthwise conv (groups = input channels = output chanels)
                                        # Choose ['kpnext'] for better kpconv
                                        # Choose ['kpmini' 'kpminix'] for depthwise kpconv
                                        # Choose ['kptran', 'kpminimod'] for kp transformer: depthwise kpconv with attention
                                        # Choose ['kpconvd', 'kpconvx'] fornew block CVPR submission
    cfg.model.shell_sizes = [1, 14, 28]
    cfg.model.kp_radius = 2.1
    cfg.model.kp_influence = 'linear'
    cfg.model.kp_aggregation = 'nearest'  # 'sum', 'nearest'
    cfg.model.conv_groups = -1   # -1 for depthwise convolution, 1 for normal conv  
    
    cfg.model.share_kp = True           #  share kernels within layers                

    cfg.data.init_sub_size = -1         # Even if we do not subsample initially, we still have to define this to define the size of convolutions 
    cfg.data.init_sub_mode = 'fps'      # Mode for initial subsampling of data
    cfg.model.in_sub_size = 0.019       # First layer subsampling sizeo optional. Try to keep a ratio of ~50 (*0.67 if fps). If negative, and fps, it is stride
    cfg.model.in_sub_mode = 'grid'      # Mode for input subsampling
    cfg.model.radius_scaling = 2.0      # We increase the convolution radius more slowly here.
    cfg.train.in_radius = -1024         # If negative, =number of points per input

    cfg.model.kpx_upcut = False         # Are we using upcuts
    cfg.model.grid_pool = True          # Are we using pure grid pooling and unpooling like PointTransformer v2
    cfg.model.decoder_layer = True      # Add a layer in decoder like PointTransformer v2
    cfg.model.upsample_n = 3            # Number of neighbors used for nearest neighbor linear interpolation (ignoeed if grid_pool)
    cfg.model.drop_path_rate = 0.3      # Rate for DropPath to make a stochastic depth model.

    # FastAdapter extension for object classification. The paper evaluates
    # segmentation; this keeps the same fixed-anchor P2A/A2P mechanism.
    cfg.model.fa_enabled = False
    cfg.model.fa_train_mode = 'joint'     # 'joint' or 'adapter_head'
    cfg.model.fa_num_anchors = 64
    cfg.model.fa_anchor_mode = 'fps'
    cfg.model.fa_anchor_level = 0
    cfg.model.fa_geometry_dim = 16
    cfg.model.fa_attention_dim = 64
    cfg.model.fa_attention_heads = 4
    cfg.model.fa_chunk_size = 4096
    cfg.model.fa_cross_layer = True
    cfg.model.fa_spatial = True
    cfg.model.fa_dropout = 0.0
    cfg.model.fa_residual_init = 1e-3

    cfg.model.input_channels = 4        # This value has to be compatible with one of the dataset input features definition
    
    # cfg.model.neighbor_limits = [8, 8, 8, 9, 9, 10, 10, 10, 10]
    # cfg.model.neighbor_limits = [8, 9, 10, 12, 12, 12]
    # cfg.model.neighbor_limits = [10, 11, 12, 14, 14]        
    # cfg.model.neighbor_limits = [16, 17, 18, 18, 18]      
    # cfg.model.neighbor_limits = [35, 40, 50, 50, 50]      # Use empty list to let calibration get the values
    # cfg.model.neighbor_limits = [16, 16, 16, 16, 16]      # List for point_transformer
    cfg.model.neighbor_limits = [12, 16, 20, 20, 20]    # List for point_transformer
    

    # Specific parameters for involution and transformers
    cfg.model.use_strided_conv = True           # Use convolution op for strided layers instead of involution
    cfg.model.first_inv_layer = 1               # Use involution layers only from this layer index (from 0 to n_layer - 1)
    cfg.model.inv_groups = 8                    # negative values to specify CpG instead of G
    cfg.model.inv_grp_norm = True
    cfg.model.inv_act = 'sigmoid'               # 'none', 'sigmoid', 'softmax', 'tanh'
    
            
    # Specific parameters for kpinv 
    cfg.model.kpinv_reduc = 1
    cfg.model.kpx_expansion = 8

    # Training parameters
    # -------------------

    # Input threads
    cfg.train.num_workers = 10
    
    # Are we using spheres/cubes/cylinders/cubic_cylinders as input
    cfg.data.use_cubes = False
    cfg.data.cylindric_input = False

    # How do we sample the input elements (spheres or cubes)
    cfg.train.data_sampler = 'c-regular'   # 'random', 'c-random', 'regular' or 'c-regular'

    # Batch related_parames
    cfg.train.batch_size = 32                 # Target batch size. If you don't want calibration, you can directly set train.batch_limit
    cfg.train.accum_batch = 2                 # Accumulate batches for an effective batch size of batch_size * accum_batch.
    cfg.train.steps_per_epoch = None
    
    # Training length
    cfg.train.max_epoch = 250
    
    # Deformations
    cfg.train.deform_loss_factor = 0.1      # Reduce to reduce influence for deformation on overall features
    cfg.train.deform_lr_factor = 1.0        # Higher so that deformation are learned faster (especially if deform_loss_factor is low)

    # Optimizer
    cfg.train.optimizer = 'AdamW'
    cfg.train.adam_b = (0.9, 0.999)
    cfg.train.adam_eps = 1e-08
    cfg.train.weight_decay = 0.01     # for KPConv
    # cfg.train.weight_decay = 0.0001     # for transformer
    cfg.train.smooth_labels = True

    # Cyclic lr 
    cfg.train.cyc_lr0 = 5e-4                # Float, Start (minimum) learning rate of 1cycle decay
    cfg.train.cyc_lr1 = 5e-3                # Float, Maximum learning rate of 1cycle decay
    cfg.train.cyc_raise_n = 1               #   Int, Raise rate for first part of 1cycle = number of epoch to multiply lr by 10
    cfg.train.cyc_decrease10 = 100           #   Int, Decrease rate for second part of 1cycle = number of epoch to divide lr by 10
    cfg.train.cyc_plateau = 1               #   Int, Number of epoch for plateau at maximum lr

    # import matplotlib.pyplot as plt
    # fig = plt.figure('lr')
    # y = [init_lr]
    # for i in range(cfg.train.max_epoch):
    #     y.append(y[-1])
    #     if str(i) in cfg.train.lr_decays:
    #         y[-1] *= cfg.train.lr_decays[str(i)]
    # plt.plot(y)
    # plt.xlabel('epochs')
    # plt.ylabel('lr')
    # plt.yscale('log')
    # ax = fig.gca()
    # ax.grid(linestyle='-.', which='both')
    # plt.show()
    # a = 1/0

    # Train Augmentations
    cfg.augment_train.anisotropic = True
    cfg.augment_train.scale = [0.9, 1.1]
    cfg.augment_train.flips = [0.5, 0, 0]
    cfg.augment_train.rotations = 'vertical'
    cfg.augment_train.jitter = 0.001
    cfg.augment_train.color_drop = 0.2
    cfg.augment_train.chromatic_contrast = True
    cfg.augment_train.chromatic_all = False
    cfg.augment_train.chromatic_norm = True
    cfg.augment_train.height_norm = False

    cfg.augment_train.rsmix_prob = 0.5 
    cfg.augment_train.rsmix_beta = 1.0 
    cfg.augment_train.rsmix_nsample = 512


    
    # Test parameters
    # ---------------

    # How do we sample the input elements (spheres or cubes)
    cfg.test.in_radius = 100.0                # For ScanObjectNN test radius or num points????
    cfg.test.data_sampler = 'regular'       # 'regular' to pick spheres regularly accross the data.

    cfg.test.max_steps_per_epoch = 100       # Size of one validation epoch (should be small)
    cfg.test.batch_limit = -1
    cfg.test.batch_size = -1

    cfg.test.val_momentum = 0.95

    # Test Augmentations
    cfg.augment_test.anisotropic = False
    cfg.augment_test.scale = [0.99, 1.01]
    cfg.augment_test.flips = [0.5, 0, 0]
    cfg.augment_test.rotations = 'vertical'
    cfg.augment_test.jitter = 0
    cfg.augment_test.color_drop = 0.0
    cfg.augment_test.chromatic_contrast = False
    cfg.augment_test.chromatic_all = False

    cfg.augment_test.rsmix_prob = 0.0
    cfg.augment_test.rsmix_beta = 0.0 
    cfg.augment_test.rsmix_nsample = 512

    return cfg


def set_seed(seed):
    """Seed the RNGs used by model initialization and data loading."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def adjust_config(cfg):

    # Model
    if cfg.model.kp_aggregation == 'nearest':
        cfg.model.kp_sigma = cfg.model.kp_radius
    else:
        cfg.model.kp_sigma = 0.7 * cfg.model.kp_radius

    # Checkpoint gap
    cfg.train.checkpoint_gap = cfg.train.max_epoch // 5

    # Learning rate
    raise_rate = (cfg.train.cyc_lr1 / cfg.train.cyc_lr0)**(1/cfg.train.cyc_raise_n)
    decrease_rate = 0.1**(1 / cfg.train.cyc_decrease10)
    cfg.train.lr = cfg.train.cyc_lr0
    cfg.train.lr_decays = {str(i): raise_rate for i in range(1, cfg.train.cyc_raise_n + 1)}
    for i in range(cfg.train.cyc_raise_n + 1 + cfg.train.cyc_plateau, cfg.train.max_epoch):
        cfg.train.lr_decays[str(i)] = decrease_rate

    # Test
    cfg.augment_test.chromatic_norm = cfg.augment_train.chromatic_norm
    cfg.augment_test.height_norm = cfg.augment_train.height_norm
    cfg.test.num_workers = cfg.train.num_workers

    return cfg


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#
if __name__ == '__main__':

    cfg = my_config()
    set_seed(cfg.exp.seed)

    # First create a tensor on GPU to signal that we use it
    device = init_gpu()
    a = torch.zeros((1,), device=device)

    ###################
    # Define parameters
    ###################

    # Add argument here to handle it
    str_args = ['model.kp_mode',
                'train.data_sampler',
                'model.kp_aggregation',
                'model.kp_influence',
                'model.norm',
                'model.inv_act',
                'model.fa_train_mode',
                'model.fa_anchor_mode']

    float_args = ['train.weight_decay',
                  'train.in_radius',
                  'augment_train.rsmix_prob',
                  'augment_train.rsmix_beta',
                  'data.init_sub_size',
                  'model.in_sub_size',
                  'model.kp_radius',
                  'model.channel_scaling',
                  'model.drop_path_rate',
                  'model.kp_sigma',
                  'model.radius_scaling',
                  'model.fa_dropout',
                  'model.fa_residual_init']

    int_args = ['model.conv_groups',
                'model.inv_groups',
                'model.init_channels',
                'model.first_inv_layer',
                'train.cyc_decrease10',
                'augment_train.rsmix_nsample',
                'model.input_channels',
                'train.max_epoch',
                'model.fa_num_anchors',
                'model.fa_anchor_level',
                'model.fa_geometry_dim',
                'model.fa_attention_dim',
                'model.fa_attention_heads',
                'model.fa_chunk_size',
                'exp.seed']

    bool_args = ['model.use_strided_conv',
                 'model.inv_grp_norm',
                 'model.kpx_upcut',
                 'data.use_cubes',
                 'data.cylindric_input',
                 'augment_train.chromatic_contrast',
                 'augment_train.chromatic_all',
                 'augment_train.chromatic_norm',
                 'model.decoder_layer',
                 'model.share_kp',
                 'model.fa_enabled',
                 'model.fa_cross_layer',
                 'model.fa_spatial',
                 'augment_train.height_norm']

    list_args = ['model.shell_sizes',
                 'model.layer_blocks',
                 'model.neighbor_limits']

    parser = argparse.ArgumentParser()
    for str_arg_name in str_args:
        parser_name = '--' + str_arg_name.split('.')[-1]
        parser.add_argument(parser_name, type=str)

    for float_arg_name in float_args:
        parser_name = '--' + float_arg_name.split('.')[-1]
        parser.add_argument(parser_name, type=float)

    for int_arg_name in int_args:
        parser_name = '--' + int_arg_name.split('.')[-1]
        parser.add_argument(parser_name, type=int)

    for bool_arg_name in bool_args:
        parser_name = '--' + bool_arg_name.split('.')[-1]
        parser.add_argument(parser_name, type=int)

    for list_arg_name in list_args:
        parser_name = '--' + list_arg_name.split('.')[-1]
        parser.add_argument(parser_name, nargs='+', type=int)

    # Log path special arg
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--log_path', type=str)
    parser.add_argument('--resume_path', type=str,
                        help='Checkpoint from which to resume an interrupted training run.')
    parser.add_argument('--finetune_path', type=str,
                        help='Backbone checkpoint for non-strict adapter/head fine-tuning.')
    args = parser.parse_args()

    resume_path = None
    finetune_path = None
    if args.resume_path is not None and args.finetune_path is not None:
        parser.error('--resume_path and --finetune_path are mutually exclusive.')

    if args.finetune_path is not None:
        finetune_path = os.path.abspath(args.finetune_path)
        if not os.path.isfile(finetune_path):
            parser.error('Checkpoint not found: {:s}'.format(finetune_path))

    if args.resume_path is not None:
        resume_path = os.path.abspath(args.resume_path)
        if not os.path.isfile(resume_path):
            parser.error('Checkpoint not found: {:s}'.format(resume_path))

        checkpoint_dir = os.path.dirname(resume_path)
        resume_log_path = os.path.dirname(checkpoint_dir)
        if os.path.basename(checkpoint_dir) != 'checkpoints':
            parser.error('Expected checkpoint to be in a checkpoints directory: {:s}'.format(resume_path))
        if args.log_path is not None and os.path.abspath(args.log_path) != resume_log_path:
            parser.error('--log_path must match the checkpoint experiment directory when resuming.')

        cfg = load_cfg(resume_log_path)
        if os.path.abspath(cfg.exp.log_dir) != resume_log_path:
            parser.error('Checkpoint and saved configuration refer to different experiment directories.')
        if args.dataset_path is not None and os.path.realpath(args.dataset_path) != os.path.realpath(cfg.data.path):
            parser.error('--dataset_path differs from the dataset recorded by the checkpoint.')

        changed_args = []
        for all_args in [str_args, float_args, int_args, list_args, bool_args]:
            for arg_name in all_args:
                key2 = arg_name.split('.')[1]
                if getattr(args, key2) is not None:
                    changed_args.append('--' + key2)
        if changed_args:
            parser.error('Training-configuration overrides are not allowed when resuming: {:s}'.format(
                ', '.join(changed_args)))

    # Load data parameters
    if resume_path is not None:
        pass
    elif args.dataset_path is not None:
        cfg.data.update(ScanObjectNN_cfg(cfg, dataset_path=args.dataset_path).data)
    else:
        cfg.data.update(ScanObjectNN_cfg(cfg).data)

    # Load experiment parameters
    if resume_path is not None:
        pass
    elif args.log_path is not None:
        get_directories(cfg, log_path=args.log_path)
    else:
        get_directories(cfg)

    # Update parameters
    if resume_path is None:
        for all_args in [str_args, float_args, int_args, list_args, bool_args]:
            for arg_name in all_args:
                key1, key2 = arg_name.split('.')
                new_arg = getattr(args, key2)
                if new_arg is not None:
                    cfg[key1][key2] = new_arg

    # Sepcial boolean handling
    if resume_path is None:
        for arg_name in bool_args:
            key1, key2 = arg_name.split('.')
            new_arg = getattr(args, key2)
            if new_arg is not None:
                cfg[key1][key2] = bool(new_arg)

    # Apply a CLI seed override before creating datasets, samplers, or models.
    set_seed(cfg.exp.seed)

    if cfg.train.in_radius < 0:
        cfg.train.in_radius = int(cfg.train.in_radius)


    # Adjust config after parameters have been changed
    cfg = adjust_config(cfg)

    
    ##############
    # Prepare Data
    ##############

    print('\n')
    frame_lines_1(['Data Preparation'])

    # Load dataset
    underline('Loading training dataset')
    training_dataset = ScanObjectNNDataset(cfg,
                                           chosen_set='training',
                                           precompute_pyramid=True)

    underline('Loading validation dataset')
    test_dataset = ScanObjectNNDataset(cfg,
                                       chosen_set='validation',
                                       precompute_pyramid=True)

    # Calib from training data
    if resume_path is None:
        training_dataset.calib_batch(cfg, update_test=True)
        training_dataset.calib_neighbors(cfg)
    test_dataset.b_n = cfg.test.batch_size
    test_dataset.b_lim = cfg.test.batch_limit

    # Save configuration now that it is complete
    if resume_path is None:
        save_cfg(cfg)
    
    # Initialize samplers
    training_sampler = ObjClassifSampler(training_dataset)
    test_sampler = ObjClassifSampler(test_dataset)

    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_size=1,
                                 sampler=training_sampler,
                                 collate_fn=ObjClassifCollate,
                                 num_workers=cfg.train.num_workers,
                                 pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=ObjClassifCollate,
                             num_workers=cfg.test.num_workers,
                             pin_memory=True)


    ###############
    # Build network
    ###############

    print()
    frame_lines_1(['Model preparation'])

    # Define network model
    underline('Loading network')
    t1 = time.time()
    modulated = False
    if 'mod' in cfg.model.kp_mode:
        modulated = True
    if cfg.model.kp_mode in ['kpconvx', 'kpconvd', 'kpconv', 'kpconvtest']:
        net = KPNeXt(cfg)
    elif cfg.model.kp_mode.startswith('kpconvold') or cfg.model.kp_mode in ['kpmini', 'kpminix']:
        net = KPConvFCNN(cfg, modulated=modulated, deformable=False)
    elif cfg.model.kp_mode.startswith('kpdef'):
        net = KPConvFCNN(cfg, modulated=modulated, deformable=True)
    elif cfg.model.kp_mode.startswith('kpinv') or cfg.model.kp_mode.startswith('kptran') or cfg.model.kp_mode.startswith('kpminimod'):
        net = KPInvFCNN(cfg)
    elif cfg.model.kp_mode.startswith('transformer') or cfg.model.kp_mode.startswith('inv_'):
        net = InvolutionFCNN(cfg)
    elif cfg.model.kp_mode.startswith('kpnext'):
        net = KPNeXt(cfg, modulated=modulated, deformable=False)


    # Show model size
    print()
    # print(net)
    print("Model size %i" % sum(param.numel() for param in net.parameters() if param.requires_grad))


    # # Debugging info
    # print('\n*************************************\n')
    # print(net.state_dict().keys())
    # print('\n*************************************\n')
    # for param in net.parameters():
    #     if param.requires_grad:
    #         print(param.shape)
    # print('\n*************************************\n')
    # print("Model size %i" % sum(param.numel() for param in net.parameters() if param.requires_grad))
    # print('\n*************************************\n')


    # Start training
    print('\n')
    frame_lines_1(['Training and Validation'])
    checkpoint_path = resume_path if resume_path is not None else finetune_path
    train_and_validate(net, training_loader, test_loader, cfg,
                       chkp_path=checkpoint_path,
                       finetune=finetune_path is not None,
                       on_gpu=True)
