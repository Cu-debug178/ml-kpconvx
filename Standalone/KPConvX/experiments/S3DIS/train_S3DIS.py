#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# ----------------------------------------------------------------------------------------------------------------------
#
#   Hugues THOMAS - 06/10/2023
#
#   KPConvX project: train_S3DIS.py
#       > Training script for S3DIS experiments
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
import signal
import argparse
import random
import gc
import numpy as np

# Set before importing torch so the CUDA allocator uses expandable segments.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

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
from utils.python_config import (
    apply_config_options,
    apply_python_config,
    PythonConfigError,
)
from utils.printing import frame_lines_1, underline
from utils.gpu_init import init_gpu
from utils.training_schedule import rebuild_cyclic_lr

from models.KPNext import KPNeXt
from models.KPConvNet import KPFCNN as KPConvFCNN
from models.KPInvNet import KPInvFCNN
from models.InvolutionNet import InvolutionFCNN

from data_handlers.scene_seg import SceneSegSampler, SceneSegCollate

# from experiments.S3DIS.S3DIS import S3DIS_cfg, S3DISDataset
from experiments.S3DIS.S3DIS_rooms import S3DIR_cfg, S3DIRDataset

from tasks.trainval import train_and_validate
from tasks.validation import get_validation_mode


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
    cfg.exp.seed = 57106803  # Match the aligned Pointcept S3DIS Area_5 baseline.

    # Network parameters
    # ------------------

    # cfg.model.layer_blocks = (3,  3,  9, 12, 3)   # KPNetX-S
    # cfg.model.layer_blocks = (4,  4, 12, 20,  4)    # KPNetX-L

    # cfg.model.layer_blocks = (2, 2, 2, 8, 2)   # KPConvX-S
    cfg.model.layer_blocks = (3, 3, 9, 12, 3)    # KPConvX-L


    cfg.model.norm = 'batch' # batch, layer
    cfg.model.init_channels = 64  # 48, 64, 80, 96
    cfg.model.channel_scaling = 1.41  # 2 or sqrt(2) or in between?

    cfg.model.kp_mode = 'kpconvx'       # Choose ['kpconv', 'kpdef', 'kpinv', 'kpinvx'].
                                        # Choose ['inv_v1', 'inv_v2', 'inv_v3', 'inv_v4', 'transformer']
                                        # Choose ['kpconv-mod', 'kpdef-mod', 'kpconv-geom'] for modulations
                                        # Choose ['kpconv-depth'] for depthwise conv (groups = input channels = output chanels)
                                        # Choose ['kpnext'] for better kpconv
                                        # Choose ['kpmini' 'kpminix'] for depthwise kpconv
                                        # Choose ['kptran', 'kpminimod'] for kp transformer: depthwise kpconv with attention
                                        # Choose ['kpconvd', 'kpconvx'] fornew block CVPR submission
    cfg.model.shell_sizes = [1, 14, 28]
    cfg.model.kp_radius = 2.1
    cfg.model.kp_influence = 'linear'       # 'constant', 'linear', 'gaussian'
    cfg.model.kp_aggregation = 'nearest'    # 'sum', 'nearest'
    cfg.model.conv_groups = -1   # -1 for depthwise convolution, 1 for normal conv  
    
    cfg.model.share_kp = True           #  share kernels within layers                

    cfg.data.init_sub_size = 0.04       # -1.0 so that dataset point clouds are not initially subsampled
    cfg.data.init_sub_mode = 'grid'     # Mode for initial subsampling of data
    cfg.model.in_sub_size = 0.04        # Adapt this with train.in_radius. Try to keep a ratio of ~50 (*0.67 if fps). If negative, and fps, it is stride
    cfg.model.in_sub_mode = 'grid'      # Mode for input subsampling
    cfg.model.radius_scaling = 2.2      # Increase conv radius by this much  


    cfg.model.kpx_upcut = False         # Are we using upcuts
    cfg.model.grid_pool = True          # Are we using pure grid pooling and unpooling like PointTransformer v2
    cfg.model.decoder_layer = True      # Add a layer in decoder like PointTransformer v2
    cfg.model.upsample_n = 3            # Number of neighbors used for nearest neighbor linear interpolation (ignoeed if grid_pool)
    cfg.model.drop_path_rate = 0.3      # Rate for DropPath to make a stochastic depth model.

    # FastAdapter P2A/A2P path for the existing grid hierarchy.
    cfg.model.fa_enabled = False
    cfg.model.fa_train_mode = 'joint'     # 'joint' or 'adapter_head'
    cfg.model.fa_num_anchors = 100
    cfg.model.fa_anchor_mode = 'fps'
    cfg.model.fa_anchor_level = 0
    cfg.model.fa_geometry_dim = 16
    cfg.model.fa_attention_dim = 64
    cfg.model.fa_attention_heads = 4
    cfg.model.fa_chunk_size = 16384
    cfg.model.fa_cross_layer = True
    cfg.model.fa_spatial = True
    cfg.model.fa_dropout = 0.0
    cfg.model.fa_residual_init = 1e-3

    # LitePT-inspired hierarchy: configured KP conv at high resolution, PointROPE token
    # attention at low resolution. Disabled by default for baseline parity.
    cfg.model.litept_enabled = False
    cfg.model.litept_conv_stages = 3
    cfg.model.litept_handover_stage = 0  # Must be litept_conv_stages + 1 when enabled.
    cfg.model.litept_patch_size = 128
    cfg.model.litept_num_heads = 8
    cfg.model.litept_attention_ratio = 1.0
    cfg.model.litept_mlp_ratio = 4.0
    cfg.model.litept_rope_base = 100.0
    cfg.model.litept_rope_enabled = True
    cfg.model.litept_attention_dropout = 0.0
    cfg.model.litept_projection_dropout = 0.0
    cfg.model.litept_orders = 'z,z-trans'
    cfg.model.litept_light_decoder = True
    cfg.model.litept_legacy_kpconvd_encoder = False

    # Kernel-to-Token Geometry Handover candidates for short warm-start screening.
    cfg.model.ktha_mode = 'none'
    cfg.model.ktha_source_stage = 3
    cfg.model.ktha_target_stages = '4'
    cfg.model.ktha_relation_dim = 8
    cfg.model.ktha_hidden_dim = 0
    cfg.model.ktha_shuffle_geometry = False
    cfg.model.ktha_signature_ablation = 'none'
    cfg.model.ktha_train_mode = 'joint'

    cfg.model.glskf_mode = 'none'
    cfg.model.glskf_refine_stage = 3
    cfg.model.glskf_context_stages = '4,5'
    cfg.model.glskf_groups = 8
    cfg.model.glskf_hidden_dim = 64
    cfg.model.glskf_matched_hidden_dim = 0
    cfg.model.glskf_detach_context = False
    cfg.model.glskf_context_control = 'none'
    cfg.model.glskf_inference_ablation = 'none'
    cfg.model.glskf_deep_residual = False
    cfg.model.glskf_train_mode = 'joint'

    cfg.model.input_channels = 5    # This value has to be compatible with one of the dataset input features definition
    
    # cfg.model.neighbor_limits = [10, 12, 12, 12, 12]      # Use empty list to let calibration get the values
    # cfg.model.neighbor_limits = [12, 14, 16, 16, 16]      # Use empty list to let calibration get the values
    # cfg.model.neighbor_limits = [16, 17, 18, 18, 18]      # Use empty list to let calibration get the values
    # cfg.model.neighbor_limits = [35, 40, 50, 50, 50]    # Use empty list to let calibration get the values
    # cfg.model.neighbor_limits = [16, 16, 16, 16, 16]    # List for point_transformer

    cfg.model.neighbor_limits = [12, 16, 20, 20, 20]    # List for point_transformer
    # cfg.model.neighbor_limits = []      # Use empty list to let calibration get the values


    # Specific parameters for involution and transformers
    cfg.model.use_strided_conv = True           # Use convolution op for strided layers instead of involution
    cfg.model.first_inv_layer = 1               # Use involution layers only from this layer index (from 0 to n_layer - 1)
    cfg.model.inv_groups = 4                    # negative values to specify CpG instead of G
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
    cfg.train.data_sampler = 'A-random'     #  'random', 'A-random' or 'c-random' for class balanced random sampling

    # Input spheres radius. Adapt this with model.in_sub_size. Try to keep a ratio of ~50
    cfg.train.in_radius = 2.1               # If negative, =number of points per input. Use negative to compare models

    # Batch related_params
    cfg.train.batch_size = 4                # Target batch size. If you don't want calibration, you can directly set train.batch_limit
    cfg.train.accum_batch = 6               # Accumulate batches for an effective batch size of batch_size * accum_batch.
    cfg.train.steps_per_epoch = 300

    # Training length
    cfg.train.max_epoch = 450  # 100
    cfg.train.checkpoint_start = 150
    cfg.train.checkpoint_gap = 10
    cfg.train.validation_mode = 'partial'
    cfg.train.save_best_val = False
    cfg.train.save_best_val_cycle = True
    cfg.train.auto_test_vote10 = True
    
    # Deformations
    cfg.train.deform_loss_factor = 0.1      # Reduce to reduce influence for deformation on overall features
    cfg.train.deform_lr_factor = 1.0        # Higher so that deformation are learned faster (especially if deform_loss_factor is low)

    # Optimizer
    cfg.train.optimizer = 'AdamW'
    cfg.train.adam_b = (0.9, 0.999)
    cfg.train.adam_eps = 1e-08
    cfg.train.weight_decay = 0.05     # for KPConv
    # cfg.train.weight_decay = 0.0001     # for transformer
    cfg.train.monitor_enabled = False
    cfg.train.monitor_interval = 50

    # Cyclic lr 
    cfg.train.cyc_lr0 = 1e-4                # Float, Start (minimum) learning rate of 1cycle decay
    cfg.train.cyc_lr1 = 5e-3                # Float, Maximum learning rate of 1cycle decay
    cfg.train.cyc_raise_n = 30              #   Int, Raise rate for first part of 1cycle = number of epoch to multiply lr by 10
    cfg.train.cyc_decrease10 = 120          #   Int, Decrease rate for second part of 1cycle = number of epoch to divide lr by 10
    cfg.train.cyc_plateau = 5               #   Int, Number of epoch for plateau at maximum lr

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
    cfg.augment_train.jitter = 0.005
    cfg.augment_train.color_drop = 0.2
    cfg.augment_train.chromatic_contrast = True
    cfg.augment_train.chromatic_all = False
    cfg.augment_train.chromatic_norm = True
    cfg.augment_train.height_norm = False
    
    cfg.augment_train.pts_drop_p = -1
    cfg.augment_train.mix3D = -1.0  # 0.8

    

    
    # Test parameters
    # ---------------

    # How do we sample the input elements (spheres or cubes)
    cfg.test.in_radius = 100.0                # For S3DIS 4 meters is very large, cover a whole part of the test set with full rooms
    cfg.test.data_sampler = 'regular'       # 'regular' to pick spheres regularly accross the data.

    cfg.test.max_steps_per_epoch = 100       # Size of one validation epoch (should be small)
    # A limit of 1 is a sentinel that stops packing after the first complete
    # room fragment; it does not truncate that fragment to one point.
    cfg.test.batch_limit = 1
    cfg.test.batch_size = 1

    cfg.test.val_momentum = 0.9

    # Test Augmentations
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


    # Input radius
    if cfg.train.in_radius > 0:

        # In case we use cubes adjust the size
        if cfg.data.use_cubes:
            if cfg.data.cylindric_input:
                cfg.train.in_radius *= np.pi**(1/2) / 2  # ratio between square and circle area
            else:
                cfg.train.in_radius *= (4 / 3 * np.pi)**(1/3) / 2  # ratio between cube and sphere volume

    else:

        # In case we sample a fixed number od points, some options cannot be used
        if cfg.train.data_sampler == 'regular':
            raise ValueError('Unable to use regular sampling with fixed number of input points. We need fixed radius.')

        # We have to used input already subsampled at the right size
        if cfg.model.in_sub_size > 0:
            cfg.data.init_sub_size = cfg.model.in_sub_size

    # Learning rate
    # This runs after CLI overrides, so compressed 250-epoch launchers rebuild
    # the schedule with their final max_epoch and cyc_decrease10 values.
    rebuild_cyclic_lr(cfg.train)

    # Test
    cfg.augment_test.chromatic_norm = cfg.augment_train.chromatic_norm
    cfg.augment_test.height_norm = cfg.augment_train.height_norm
    cfg.test.num_workers = cfg.train.num_workers

    return cfg


def configure_validation_mode(cfg):
    """Apply the deterministic full-room validation protocol when requested."""

    mode = get_validation_mode(cfg)
    if mode != 'full_identity':
        return cfg

    # Identity validation uses every regular room center exactly once per epoch.
    cfg.test.data_sampler = 'regular'
    cfg.test.in_radius = 100.0
    # One deterministic pass over every regular room center.  batch_limit=1
    # packs one complete room fragment per item, so sampler size equals the
    # number of covered room centers exactly.
    cfg.test.batch_limit = 1
    cfg.test.batch_size = 1
    cfg.test.max_steps_per_epoch = 9999999
    cfg.test.val_momentum = 0.0

    cfg.augment_test.anisotropic = False
    cfg.augment_test.scale = [1.0, 1.0]
    cfg.augment_test.flips = [0.0, 0.0, 0.0]
    cfg.augment_test.rotations = 'none'
    cfg.augment_test.jitter = 0.0
    cfg.augment_test.color_drop = 0.0
    cfg.augment_test.chromatic_contrast = False
    cfg.augment_test.chromatic_all = False
    cfg.augment_test.pts_drop_p = -1.0
    return cfg


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#
if __name__ == '__main__':

    cfg = my_config()

    ###################
    # Define parameters
    ###################

    # Add argument here to handle it
    str_args = ['model.kp_mode',
                'train.data_sampler',
                'train.amp_dtype',
                'train.validation_mode',
                'model.kp_aggregation',
                'model.kp_influence',
                'model.norm',
                'model.inv_act',
                'model.fa_train_mode',
                'model.fa_anchor_mode',
                'model.litept_orders',
                'model.ktha_mode',
                'model.ktha_target_stages',
                'model.ktha_signature_ablation',
                'model.ktha_train_mode',
                'model.glskf_mode',
                'model.glskf_context_stages',
                'model.glskf_context_control',
                'model.glskf_inference_ablation',
                'model.glskf_train_mode']

    float_args = ['train.weight_decay',
                  'train.in_radius',
                  'data.init_sub_size',
                  'model.in_sub_size',
                  'model.kp_radius',
                  'model.channel_scaling',
                  'model.drop_path_rate',
                  'model.kp_sigma',
                  'model.radius_scaling',
                  'model.fa_dropout',
                  'model.fa_residual_init',
                  'model.litept_attention_ratio',
                  'model.litept_mlp_ratio',
                  'model.litept_rope_base',
                  'model.litept_attention_dropout',
                  'model.litept_projection_dropout',
                  'augment_train.mix3D']

    int_args = ['model.conv_groups',
                'model.inv_groups',
                'model.init_channels',
                'model.first_inv_layer',
                'train.batch_size',
                'train.accum_batch',
                'train.cyc_decrease10',
                'train.max_epoch',
                'train.checkpoint_start',
                'train.checkpoint_gap',
                'train.monitor_interval',
                'model.fa_num_anchors',
                'model.fa_anchor_level',
                'model.fa_geometry_dim',
                'model.fa_attention_dim',
                'model.fa_attention_heads',
                'model.fa_chunk_size',
                'model.litept_conv_stages',
                'model.litept_handover_stage',
                 'model.litept_patch_size',
                 'model.litept_num_heads',
                 'model.ktha_source_stage',
                 'model.ktha_relation_dim',
                 'model.ktha_hidden_dim',
                 'model.glskf_refine_stage',
                 'model.glskf_groups',
                 'model.glskf_hidden_dim',
                 'model.glskf_matched_hidden_dim',
                 'exp.seed']

    bool_args = ['train.monitor_enabled',
                 'train.amp_enabled',
                 'train.save_best_val',
                 'train.save_latest_val',
                 'train.save_best_val_cycle',
                 'train.save_fraction_checkpoints',
                 'train.save_periodic_checkpoints',
                 'train.auto_test_vote10',
                 'model.use_strided_conv',
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
                 'model.litept_enabled',
                 'model.litept_rope_enabled',
                 'model.litept_light_decoder',
                 'model.litept_legacy_kpconvd_encoder',
                 'model.ktha_shuffle_geometry',
                 'model.glskf_detach_context',
                 'model.glskf_deep_residual',
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
    parser.add_argument('--config', '--config-file', dest='config', type=str,
                        help='Trusted Python override, relative to KPConvX/configs or the current directory.')
    parser.add_argument('--options', nargs='+', metavar='SECTION.KEY=VALUE',
                        help='Generic config overrides applied after the Python config and before named flags.')
    parser.add_argument('--resume_path', type=str,
                        help='Checkpoint from which to resume an interrupted training run.')
    parser.add_argument('--finetune_path', type=str,
                        help='Backbone checkpoint for non-strict adapter/head fine-tuning.')
    args = parser.parse_args()

    resume_path = None
    finetune_path = None
    if args.resume_path is not None and args.finetune_path is not None:
        parser.error('--resume_path and --finetune_path are mutually exclusive.')
    if args.resume_path is not None and args.config is not None:
        parser.error('--config is not allowed with --resume_path; the saved parameters.json must be used.')
    if args.resume_path is not None and args.options is not None:
        parser.error('--options is not allowed with --resume_path; the saved parameters.json must be used.')

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

        # Keep the original data and experiment paths through the normal setup.
        args.dataset_path = cfg.data.path
        args.log_path = cfg.exp.log_dir
        set_seed(cfg.exp.seed)

    if resume_path is None and args.config is not None:
        try:
            cfg, config_path = apply_python_config(cfg, args.config)
        except PythonConfigError as error:
            parser.error(str(error))
        print('Using Python config: {:s}'.format(config_path))

    if resume_path is None and args.options is not None:
        try:
            cfg = apply_config_options(cfg, args.options)
        except PythonConfigError as error:
            parser.error(str(error))

    # Load data parameters
    if args.dataset_path is not None:
        cfg.data.update(S3DIR_cfg(cfg, dataset_path=args.dataset_path).data)
    else:
        cfg.data.update(S3DIR_cfg(cfg).data)

    # Load experiment parameters
    if args.log_path is not None:
        get_directories(cfg, log_path=args.log_path)
    else:
        get_directories(cfg)

    # Update parameters
    for all_args in [str_args, float_args, int_args, list_args, bool_args]:
        for arg_name in all_args:
            key1, key2 = arg_name.split('.')
            new_arg = getattr(args, key2)
            if new_arg is not None:
                cfg[key1][key2] = new_arg

    # Sepcial boolean handling
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
    cfg = configure_validation_mode(cfg)

    # Claim the GPU only after all configuration inputs have been validated.
    device = init_gpu()
    a = torch.zeros((1,), device=device)

    
    ##############
    # Prepare Data
    ##############

    print('\n')
    frame_lines_1(['Data Preparation'])

    # Load dataset
    underline('Loading training dataset')
    training_dataset = S3DIRDataset(cfg,
                                    chosen_set='training',
                                    precompute_pyramid=True)
    underline('Loading validation dataset')
    test_dataset = S3DIRDataset(cfg,
                                chosen_set='validation',
                                precompute_pyramid=True)
    
    # Calibration mutates batch_limit and neighbor_limits. A resumed run must reuse
    # the values recorded by the original experiment.
    if resume_path is None:
        training_dataset.calib_batch(cfg, update_test=False)
        training_dataset.calib_neighbors(cfg)
    test_dataset.b_n = cfg.test.batch_size
    test_dataset.b_lim = cfg.test.batch_limit

    # Do not rewrite the original configuration record when resuming.
    if resume_path is None:
        save_cfg(cfg)
    
    # Initialize samplers
    training_sampler = SceneSegSampler(training_dataset)
    test_sampler = SceneSegSampler(test_dataset)
    if cfg.train.validation_mode == 'full_identity':
        # Override the legacy partial-validation cap (34%-67% of the rooms).
        test_sampler.N = test_dataset.get_reg_sampling_size()

    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_size=1,
                                 sampler=training_sampler,
                                 collate_fn=SceneSegCollate,
                                 num_workers=cfg.train.num_workers,
                                 persistent_workers=False,
                                 pin_memory=True,
                                 prefetch_factor=4)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=SceneSegCollate,
                             num_workers=cfg.test.num_workers,
                             persistent_workers=False,
                             pin_memory=True,
                             prefetch_factor=2)


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


    # Model variants consume different RNG values during initialization. Reset
    # the streams before the first loader iterator so warm-start controls share
    # identical worker/data sampling streams.
    set_seed(cfg.exp.seed)

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
    train_summary = train_and_validate(
        net,
        training_loader,
        test_loader,
        cfg,
        chkp_path=checkpoint_path,
        finetune=finetune_path is not None,
        on_gpu=True,
    )

    # Launch the existing standard 10-vote evaluation only after all epochs finish.
    if cfg.train.auto_test_vote10 and train_summary.completed:
        chosen_weight = None
        if cfg.train.save_best_val_cycle:
            chosen_weight = train_summary.best_cycle_checkpoint
        if chosen_weight is None and cfg.train.save_best_val:
            chosen_weight = train_summary.best_checkpoint
        if chosen_weight is None and cfg.train.save_latest_val:
            chosen_weight = train_summary.latest_checkpoint
        if chosen_weight is None:
            raise RuntimeError(
                'auto_test_vote10 requires a selected best-cycle, best-validation, '
                'or latest checkpoint; enable one of the corresponding save switches'
            )

        del a, training_loader, test_loader, training_dataset, test_dataset, net
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from experiments.S3DIS.test_S3DIS import configure_vote_test, test_S3DIS_log

        configure_vote_test(cfg, votes=10)
        test_S3DIS_log(
            cfg.exp.log_dir,
            cfg,
            weight_path=chosen_weight,
        )
