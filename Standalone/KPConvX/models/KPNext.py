#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# ----------------------------------------------------------------------------------------------------------------------
#
#   Hugues THOMAS - 06/10/2023
#
#   KPConvX project: KPNext.py
#       > Define the network architecture for KPConvX
#

import time
import torch
import torch.nn as nn
import numpy as np

from models.generic_blocks import LinearUpsampleBlock, NearestUpsampleBlock, UnaryBlock, local_nearest_pool, GlobalAverageBlock, MaxPoolBlock, SmoothCrossEntropyLoss
from models.kpconv_blocks import KPConvBlock, KPConvResidualBlock, KPConvInvertedBlock
from models.kpnext_blocks import KPNextResidualBlock, KPNextInvertedBlock, KPNextMultiShortcutBlock, KPNextBlock
from models.fast_adapter import FastAdapterStack
from models.litept_blocks import (LiteHandoverBlock, LitePointTransformerBlock,
                                  SerializedPatchCache, parse_serialization_orders)

from utils.torch_pyramid import fill_pyramid

class KPNeXt(nn.Module):

    def __init__(self, cfg):
        """
        Class defining KPNeXt, a modern architecture inspired from ConvNext.
        Standard drop_path_rate: 0
        Standard layer_scale_init_value: 1e-6
        Standard head_init_scale: 1

        Args:
            cfg (EasyDict): configuration dictionary
        """
        super(KPNeXt, self).__init__()

        ############
        # Parameters
        ############

        # Parameters
        self.subsample_size = cfg.model.in_sub_size
        if self.subsample_size < 0:
            self.subsample_size = cfg.data.init_sub_size
        self.in_sub_mode = cfg.model.in_sub_mode
        self.kp_radius = cfg.model.kp_radius
        self.kp_sigma = cfg.model.kp_sigma
        self.neighbor_limits = cfg.model.neighbor_limits
        if cfg.model.in_sub_size > cfg.data.init_sub_size * 1.01:
            self.first_radius = cfg.model.in_sub_size * cfg.model.kp_radius
        else:
            self.first_radius = cfg.data.init_sub_size * cfg.model.kp_radius
        self.radius_scaling = cfg.model.radius_scaling
        self.first_sigma = cfg.data.init_sub_size * self.kp_sigma

        self.layer_blocks = cfg.model.layer_blocks
        self.num_layers = len(self.layer_blocks)
        self.upsample_n = cfg.model.upsample_n
        self.share_kp = cfg.model.share_kp
        self.kp_mode = cfg.model.kp_mode
        self.task = cfg.data.task
        self.grid_pool = cfg.model.grid_pool
        self.add_decoder_layer = cfg.model.decoder_layer

        # LitePT-inspired stage specialization.  This is intentionally separate
        # from KPConvX kernel attention: late stages use token self-attention.
        self.litept_enabled = bool(getattr(cfg.model, 'litept_enabled', False))
        self.litept_conv_stages = int(getattr(cfg.model, 'litept_conv_stages', 3))
        self.litept_handover_stage = int(getattr(cfg.model, 'litept_handover_stage', 0))
        self.litept_patch_size = int(getattr(cfg.model, 'litept_patch_size', 128))
        self.litept_num_heads = int(getattr(cfg.model, 'litept_num_heads', 8))
        self.litept_attention_ratio = float(getattr(cfg.model, 'litept_attention_ratio', 1.0))
        self.litept_mlp_ratio = float(getattr(cfg.model, 'litept_mlp_ratio', 4.0))
        self.litept_rope_base = float(getattr(cfg.model, 'litept_rope_base', 100.0))
        self.litept_rope_enabled = bool(getattr(cfg.model, 'litept_rope_enabled', True))
        self.litept_attention_dropout = float(getattr(cfg.model, 'litept_attention_dropout', 0.0))
        self.litept_projection_dropout = float(getattr(cfg.model, 'litept_projection_dropout', 0.0))
        self.litept_orders = parse_serialization_orders(
            getattr(cfg.model, 'litept_orders', 'z,z-trans')
        )
        self.litept_light_decoder = bool(
            getattr(cfg.model, 'litept_light_decoder', False)
        )
        if self.litept_enabled:
            if self.kp_mode not in {'kpconvd', 'kpconvx'}:
                raise ValueError(
                    "LitePT stage specialization currently requires kp_mode "
                    "'kpconvd' or 'kpconvx'."
                )
            if not 0 <= self.litept_conv_stages <= self.num_layers:
                raise ValueError('litept_conv_stages must be between 0 and num_layers')
            if not 0 <= self.litept_handover_stage <= self.num_layers:
                raise ValueError('litept_handover_stage must be 0 or a valid 1-based stage')
            if (
                self.litept_handover_stage > 0
                and self.litept_handover_stage != self.litept_conv_stages + 1
            ):
                raise ValueError(
                    'litept_handover_stage must immediately follow the '
                    'convolution-only stages (handover_stage = conv_stages + 1)'
                )
            if self.litept_patch_size < 1:
                raise ValueError('litept_patch_size must be positive')
            if self.litept_light_decoder and self.task == 'cloud_segmentation':
                self.add_decoder_layer = False

        # This context path is independent of the pyramid sampling method.
        self.fa_enabled = bool(getattr(cfg.model, 'fa_enabled', False))
        self.fa_train_mode = str(getattr(cfg.model, 'fa_train_mode', 'joint')).lower()

        # Stochastic depth decay rule
        dpr_list = np.linspace(0, cfg.model.drop_path_rate, sum(self.layer_blocks)) 
        
        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort([c for c in cfg.data.label_values if c not in cfg.data.ignored_labels])
        self.num_logits = len(self.valid_labels)

        # Variables
        in_C = cfg.model.input_channels
        first_C = cfg.model.init_channels
        conv_r = self.first_radius
        conv_sig = self.first_sigma
        channel_scaling = 2
        if 'channel_scaling' in cfg.model:
            channel_scaling = cfg.model.channel_scaling

        # Get channels at each layer
        layer_C = []
        for l in range(self.num_layers):
            target_C = first_C * channel_scaling ** l                   # Scale channels
            layer_C.append(int(np.ceil((target_C - 0.1) / 16)) * 16)    # Ensure it is divisible by 16 (even the first one)

        # Grid pooling expands the final block before pooling, so use the
        # actual feature width at each adapter insertion point.
        adapter_channels = [
            layer_C[l + 1] if self.grid_pool and l < self.num_layers - 1 else layer_C[l]
            for l in range(self.num_layers)
        ]
        if self.fa_enabled:
            self.fast_adapter = FastAdapterStack(adapter_channels, cfg.model)
        else:
            self.fast_adapter = None

        # Verify the architecture validity
        if self.layer_blocks[0] < 1:
            raise ValueError('First layer must contain at least 1 convolutional layers')
        if np.min(self.layer_blocks) < 1:
            raise ValueError('Each layer must contain at least 1 convolutional layers')
        
        #####################
        # List Encoder blocks
        #####################

        # ------ Layers 1 ------
        self._litept_patch_caches = {}
        if cfg.model.share_kp:
            self.shared_kp = [{} for _ in range(self.num_layers)]
        else:
            self.shared_kp = [None for _ in range(self.num_layers)]

        # Initial convolution or MLP
        C = layer_C[0]
        self.stem = self.get_conv_block(in_C, C, conv_r, conv_sig, cfg)
        # self.stem = self.get_unary_block(in_C, C, cfg)

        # Next blocks
        self.encoder_1 = nn.ModuleList()
        use_conv = cfg.model.first_inv_layer >= 1
        stage_kind = self._encoder_stage_kind(1, use_conv)
        for block_i in range(self.layer_blocks[0]):
            Cout = layer_C[1] if self.grid_pool and block_i == self.layer_blocks[0] - 1 else C
            self.encoder_1.append(self.get_encoder_block(
                C, Cout, conv_r, conv_sig, cfg, layer=1, block_i=block_i,
                shared_kp_data=self.shared_kp[0], stage_kind=stage_kind,
                drop_path=dpr_list[block_i]))

        # Pooling block
        self.pooling_1 = self.get_pooling_block(
            C, layer_C[1], conv_r, conv_sig, cfg,
            use_mod=(not use_conv) if not self.litept_enabled else False)

        # ------ Layers [2, 3, 4, 5] ------
        for layer in range(2, self.num_layers + 1):
            l = layer - 1

            # Update features, radius, sigma for this layer
            C = layer_C[l]
            conv_r *= self.radius_scaling
            conv_sig *= self.radius_scaling

            # Layer blocks
            use_conv = cfg.model.first_inv_layer >= layer
            stage_kind = self._encoder_stage_kind(layer, use_conv)
            encoder_i = nn.ModuleList()
            for block_i in range(self.layer_blocks[l]):
                Cout = layer_C[l+1] if self.grid_pool and layer < self.num_layers and block_i == self.layer_blocks[l] - 1 else C
                global_block_i = sum(self.layer_blocks[:l]) + block_i
                encoder_i.append(self.get_encoder_block(
                    C, Cout, conv_r, conv_sig, cfg, layer=layer, block_i=global_block_i,
                    shared_kp_data=self.shared_kp[l], stage_kind=stage_kind,
                    drop_path=dpr_list[global_block_i]))
            setattr(self, 'encoder_{:d}'.format(layer), encoder_i)

            # Pooling block (not for the last layer)
            if layer < self.num_layers:
                pooling_i = self.get_pooling_block(
                    C, layer_C[l+1], conv_r, conv_sig, cfg,
                    use_mod=(not use_conv) if not self.litept_enabled else False)
                setattr(self, 'pooling_{:d}'.format(layer), pooling_i)

        #####################
        # List Decoder blocks
        #####################

        if cfg.data.task == 'classification':

            #  ------ Head ------

            # Global pooling
            self.global_pooling = GlobalAverageBlock()

            # New head
            self.head = nn.Sequential(self.get_unary_block(layer_C[-1], 256, cfg, norm_type='none'),
                                      nn.Dropout(0.4),
                                      nn.Linear(256, self.num_logits))

            # # Old head
            # self.head = nn.Sequential(self.get_unary_block(layer_C[-1], layer_C[-1], cfg),
            #                           nn.Linear(layer_C[-1], self.num_logits))

        elif cfg.data.task == 'cloud_segmentation':

            # ------ Layers [4, 3, 2, 1] ------
            for layer in range(self.num_layers - 1, 0, -1):

                # Upsample block
                if self.grid_pool:
                    upsampling_i = NearestUpsampleBlock()
                else:
                    upsampling_i = LinearUpsampleBlock(self.upsample_n)
                setattr(self, 'upsampling_{:d}'.format(layer), upsampling_i)

                # Network layers in decoder
                C = layer_C[layer - 1]
                C1 = layer_C[layer]
                if self.grid_pool:
                    Cin = C1 + C1
                else:
                    Cin = C + C1
                decoder_norm = (
                    'layer'
                    if self.litept_enabled and self.litept_light_decoder
                    else None
                )
                decoder_unary_i = self.get_unary_block(
                    Cin, C, cfg, norm_type=decoder_norm
                )
                setattr(self, 'decoder_unary_{:d}'.format(layer), decoder_unary_i)

                # Additionnal network layer (optional)
                if self.add_decoder_layer:
                    conv_r *= 1 / self.radius_scaling
                    conv_sig *= 1 / self.radius_scaling
                    decoder_layer_i = self.get_residual_block(C, C, conv_r, conv_sig, cfg,
                                                              shared_kp_data=self.shared_kp[layer - 1])
                    setattr(self, 'decoder_layer_{:d}'.format(layer), decoder_layer_i)


            #  ------ Head ------
            
            # New head
            self.head = nn.Sequential(self.get_unary_block(layer_C[0], layer_C[0], cfg),
                                    nn.Linear(layer_C[0], self.num_logits))
            # Easy KPConv Head
            # self.head = nn.Sequential(nn.Linear(layer_C[0] * 2, layer_C[0]),
            #                           nn.GroupNorm(8, layer_C[0]),
            #                           nn.ReLU(),
            #                           nn.Linear(layer_C[0], self.num_logits))

            # My old head
            # self.head = nn.Sequential(self.get_unary_block(layer_C[0] * 2, layer_C[0], cfg, norm_type='none'),
            #                           nn.Linear(layer_C[0], self.num_logits))



        ################
        # Network Losses
        ################

        # Choose between normal cross entropy and smoothed labels
        if cfg.train.smooth_labels:
            CrossEntropy = SmoothCrossEntropyLoss
        else:
            CrossEntropy = torch.nn.CrossEntropyLoss

        if cfg.data.task == 'classification':
            self.criterion = CrossEntropy()
            
        elif cfg.data.task == 'cloud_segmentation':
            if len(cfg.train.class_w) > 0:
                class_w = torch.from_numpy(np.array(cfg.train.class_w, dtype=np.float32))
                self.criterion = CrossEntropy(weight=class_w, ignore_index=-1)
            else:
                self.criterion = CrossEntropy(ignore_index=-1)

        # self.deform_fitting_mode = config.deform_fitting_mode
        # self.deform_fitting_power = config.deform_fitting_power
        # self.deform_lr_factor = config.deform_lr_factor
        # self.repulse_extent = config.repulse_extent

        self.deform_loss_factor = cfg.train.deform_loss_factor
        self.fit_rep_ratio = cfg.train.deform_fit_rep_ratio
        self.output_loss = 0
        self.deform_loss = 0
        self.l1 = nn.L1Loss()

        self._configure_fast_adapter_training()
        return

    def _configure_fast_adapter_training(self):
        """Configure joint training or adapter/head-only fine-tuning."""

        if not self.fa_enabled or self.fa_train_mode == 'joint':
            return
        if self.fa_train_mode != 'adapter_head':
            raise ValueError(
                "model.fa_train_mode must be either 'joint' or 'adapter_head'"
            )

        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.fast_adapter.parameters():
            parameter.requires_grad = True
        for parameter in self.head.parameters():
            parameter.requires_grad = True

    def train(self, mode=True):
        """Keep the frozen backbone, including BN statistics, in eval mode."""

        super().train(mode)
        if mode and self.fa_enabled and self.fa_train_mode == 'adapter_head':
            for child_name, child_module in self.named_children():
                if child_name not in {'fast_adapter', 'head'}:
                    child_module.eval()
            self.fast_adapter.train(True)
            self.head.train(True)
        return self

    def _encoder_stage_kind(self, layer, original_use_conv):
        """Return ``conv``, ``attention``, ``handover`` or ``kpconvx``."""

        if not self.litept_enabled:
            return 'conv' if original_use_conv else 'kpconvx'
        if layer == self.litept_handover_stage:
            return 'handover'
        if layer <= self.litept_conv_stages:
            return 'conv'
        return 'attention'

    def get_encoder_block(self, in_C, out_C, radius, sigma, cfg, layer, block_i,
                          shared_kp_data, stage_kind, drop_path):
        """Build a stage-tailored encoder block.

        Early ``conv`` stages use KPConvD (kernel attention disabled).  Late
        ``attention`` stages use serialized PointROPE token attention.  A
        ``handover`` stage applies both operators sequentially.
        """

        if stage_kind in {'conv', 'kpconvx'}:
            return self.get_residual_block(
                in_C, out_C, radius, sigma, cfg,
                shared_kp_data=shared_kp_data,
                conv_layer=(stage_kind == 'conv'),
                drop_path=drop_path)

        voxel_size = max(
            float(self.subsample_size) * self.radius_scaling ** (layer - 1),
            1e-6,
        )
        order = self.litept_orders[block_i % len(self.litept_orders)]
        patch_cache = self._litept_patch_caches.setdefault(
            layer, SerializedPatchCache()
        )

        if stage_kind == 'attention':
            return LitePointTransformerBlock(
                in_channels=in_C,
                out_channels=out_C,
                voxel_size=voxel_size,
                num_heads=self.litept_num_heads,
                patch_size=self.litept_patch_size,
                attention_ratio=self.litept_attention_ratio,
                mlp_ratio=self.litept_mlp_ratio,
                rope_base=self.litept_rope_base,
                rope_enabled=self.litept_rope_enabled,
                attention_dropout=self.litept_attention_dropout,
                projection_dropout=self.litept_projection_dropout,
                drop_path=drop_path,
                order=order,
                patch_cache=patch_cache,
            )

        if stage_kind == 'handover':
            convolution = self.get_residual_block(
                in_C, out_C, radius, sigma, cfg,
                shared_kp_data=shared_kp_data,
                conv_layer=True,
                drop_path=drop_path)
            attention = LitePointTransformerBlock(
                in_channels=out_C,
                out_channels=out_C,
                voxel_size=voxel_size,
                num_heads=self.litept_num_heads,
                patch_size=self.litept_patch_size,
                attention_ratio=self.litept_attention_ratio,
                mlp_ratio=self.litept_mlp_ratio,
                rope_base=self.litept_rope_base,
                rope_enabled=self.litept_rope_enabled,
                attention_dropout=self.litept_attention_dropout,
                projection_dropout=self.litept_projection_dropout,
                drop_path=drop_path,
                order=order,
                patch_cache=patch_cache,
            )
            return LiteHandoverBlock(convolution, attention)

        raise ValueError('Unknown encoder stage kind: {:s}'.format(stage_kind))

    def get_unary_block(self, in_C, out_C, cfg, norm_type=None):

        if norm_type is None:
            norm_type = cfg.model.norm

        return UnaryBlock(in_C,
                          out_C,
                          norm_type=norm_type,
                          bn_momentum=cfg.model.bn_momentum)

    def get_conv_block(self, in_C, out_C, radius, sigma, cfg):

        # First layer is the most simple convolution possible
        return KPConvBlock(in_C,
                           out_C,
                           cfg.model.shell_sizes,
                           radius,
                           sigma,
                           influence_mode=cfg.model.kp_influence,
                           aggregation_mode=cfg.model.kp_aggregation,
                           dimension=cfg.data.dim,
                           norm_type=cfg.model.norm,
                           bn_momentum=cfg.model.bn_momentum)

    def get_pooling_block(self, in_C, out_C, radius, sigma, cfg, use_mod=False):

        if self.grid_pool:
            return MaxPoolBlock()

        else:
            # Depthwise conv 
            if cfg.model.use_strided_conv:
                return KPConvBlock(in_C,
                                out_C,
                                cfg.model.shell_sizes,
                                radius,
                                sigma,
                                influence_mode=cfg.model.kp_influence,
                                aggregation_mode=cfg.model.kp_aggregation,
                                dimension=cfg.data.dim,
                                norm_type=cfg.model.norm,
                                bn_momentum=cfg.model.bn_momentum)

            else:
                attention_groups = cfg.model.inv_groups
                if 'kpconvd' in self.kp_mode or not use_mod:
                    attention_groups = 0
                return KPNextBlock(in_C,
                                out_C,
                                cfg.model.shell_sizes,
                                radius,
                                sigma,
                                attention_groups=attention_groups,
                                attention_act=cfg.model.inv_act,
                                mod_grp_norm=cfg.model.inv_grp_norm,
                                influence_mode=cfg.model.kp_influence,
                                dimension=cfg.data.dim,
                                norm_type=cfg.model.norm,
                                bn_momentum=cfg.model.bn_momentum)
                           
    def get_residual_block(self, in_C, out_C, radius, sigma, cfg, shared_kp_data=None, 
                           conv_layer=False, drop_path=-1):

        attention_groups = cfg.model.inv_groups
        if conv_layer or 'kpconvd' in self.kp_mode:
            attention_groups = 0

        #TMP to get 
        if self.kp_mode == 'kpconv':

            return KPConvResidualBlock(in_C,
                                       out_C,
                                       cfg.model.shell_sizes,
                                       radius,
                                       sigma,
                                       groups=cfg.model.conv_groups,
                                       shared_kp_data=shared_kp_data,
                                       influence_mode=cfg.model.kp_influence,
                                       dimension=cfg.data.dim,
                                       norm_type=cfg.model.norm,
                                       bn_momentum=cfg.model.bn_momentum)
        elif self.kp_mode == 'kpconvtest':
            return KPNextResidualBlock(in_C,
                                       out_C,
                                       cfg.model.shell_sizes,
                                       radius,
                                       sigma,
                                       attention_groups=attention_groups,
                                       attention_act=cfg.model.inv_act,
                                       mod_grp_norm=cfg.model.inv_grp_norm,
                                       shared_kp_data=shared_kp_data,
                                       influence_mode=cfg.model.kp_influence,
                                       dimension=cfg.data.dim,
                                       norm_type=cfg.model.norm,
                                       bn_momentum=cfg.model.bn_momentum)

        else:
            return KPNextMultiShortcutBlock(in_C,
                                            out_C,
                                            cfg.model.shell_sizes,
                                            radius,
                                            sigma,
                                            attention_groups=attention_groups,
                                            attention_act=cfg.model.inv_act,
                                            mod_grp_norm=cfg.model.inv_grp_norm,
                                            expansion=4,
                                            drop_path_p=drop_path,
                                            layer_scale_init_v=-1.,
                                            use_upcut=cfg.model.kpx_upcut,
                                            shared_kp_data=shared_kp_data,
                                            influence_mode=cfg.model.kp_influence,
                                            dimension=cfg.data.dim,
                                            norm_type=cfg.model.norm,
                                            bn_momentum=cfg.model.bn_momentum)






    def litept_serialization_profile(self):
        """Return serialization timings for the most recent forward pass."""

        stages = {
            int(stage): cache.profile_stats
            for stage, cache in sorted(self._litept_patch_caches.items())
        }
        return {
            "stages": stages,
            "quantization_count": sum(v["quantization_count"] for v in stages.values()),
            "layout_count": sum(v["layout_count"] for v in stages.values()),
            "quantization_ms": sum(v["quantization_ms"] for v in stages.values()),
            "layout_ms": sum(v["layout_ms"] for v in stages.values()),
            "total_ms": sum(v["total_ms"] for v in stages.values()),
        }

    def forward(self, batch, verbose=False):

        # Serialization is shared by all attention blocks in a stage, but must
        # be rebuilt for each new batch because point coordinates change.
        for patch_cache in self._litept_patch_caches.values():
            patch_cache.clear()

        #  ------ Init ------
        
        if verbose:
            torch.cuda.synchronize(batch.device())
            t = [time.time()]

        # First complete the input pyramid if not already done
        if len(batch.in_dict.neighbors) < 1:
            fill_pyramid(batch.in_dict,
                         self.num_layers,
                         self.subsample_size,
                         self.first_radius,
                         self.radius_scaling,
                         self.neighbor_limits,
                         self.upsample_n,
                         sub_mode=self.in_sub_mode,
                         grid_pool_mode=self.grid_pool)

        if verbose:
            torch.cuda.synchronize(batch.device())                           
            t += [time.time()]

        # Fixed anchors are selected once and reused at every encoder stage.
        fa_state = None
        if self.fast_adapter is not None:
            fa_state = self.fast_adapter.initialize_state(
                batch.in_dict.points,
                batch.in_dict.lengths,
            )

        # Get input features
        feats = batch.in_dict.features.clone().detach()
        
        if verbose:      
            torch.cuda.synchronize(batch.device())                        
            t += [time.time()]

        
        #  ------ Stem ------
        feats = self.stem(batch.in_dict.points[0], batch.in_dict.points[0], feats, batch.in_dict.neighbors[0])
        # feats = self.stem(feats)


        #  ------ Encoder ------

        skip_feats = []
        for layer in range(1, self.num_layers + 1):

            # Get layer blocks
            l = layer -1
            block_list = getattr(self, 'encoder_{:d}'.format(layer))
            
            # Layer blocks
            if self.kp_mode in ['kpconv', 'kpconvtest']:
                for block in block_list:
                    feats = block(batch.in_dict.points[l], batch.in_dict.points[l], feats, batch.in_dict.neighbors[l])
            else:
                upcut = None
                for block in block_list:
                    feats, upcut = block(batch.in_dict.points[l], batch.in_dict.points[l], feats, batch.in_dict.neighbors[l], batch.in_dict.lengths[l], upcut=upcut)

            # Compensate geometry before skip storage and downsampling.
            if self.fast_adapter is not None:
                feats, fa_state = self.fast_adapter.forward_layer(
                    l,
                    batch.in_dict.points[l],
                    batch.in_dict.lengths[l],
                    feats,
                    fa_state,
                )
                
            if layer < self.num_layers:

                # Skip features
                skip_feats.append(feats)

                # Pooling
                layer_pool = getattr(self, 'pooling_{:d}'.format(layer))
                if self.grid_pool:
                    feats = layer_pool(feats, batch.in_dict.pools[l])
                else:
                    feats = layer_pool(batch.in_dict.points[l+1], batch.in_dict.points[l], feats, batch.in_dict.pools[l])

         
        if verbose:    
            torch.cuda.synchronize(batch.device())                         
            t += [time.time()]

        if self.task == 'classification':
            
            # Global pooling
            feats = self.global_pooling(feats, batch.in_dict.lengths[-1])

            
        elif self.task == 'cloud_segmentation':

            #  ------ Decoder ------

            for layer in range(self.num_layers - 1, 0, -1):

                # Get layer blocks
                l = layer -1    # 3, 2, 1, 0
                upsample = getattr(self, 'upsampling_{:d}'.format(layer))

                # Upsample
                if self.grid_pool:
                    feats = upsample(feats, batch.in_dict.upsamples[l])
                else:
                    feats = upsample(feats, batch.in_dict.upsamples[l], batch.in_dict.up_distances[l])

                # Concat with skip features
                feats = torch.cat([feats, skip_feats[l]], dim=1)
                
                # MLP
                unary = getattr(self, 'decoder_unary_{:d}'.format(layer))
                feats = unary(feats)

                # Optional Decoder layers
                if self.add_decoder_layer:
                    block = getattr(self, 'decoder_layer_{:d}'.format(layer))
                    if self.kp_mode in ['kpconv', 'kpconvtest']:
                        feats = block(batch.in_dict.points[l], batch.in_dict.points[l], feats, batch.in_dict.neighbors[l])
                    else:
                        feats, _ = block(batch.in_dict.points[l], batch.in_dict.points[l], feats, batch.in_dict.neighbors[l], batch.in_dict.lengths[l])

        #  ------ Head ------

        logits = self.head(feats)
                

        if verbose:
            torch.cuda.synchronize(batch.device())                      
            t += [time.time()]
            mean_dt = 1000 * (np.array(t[1:]) - np.array(t[:-1]))
            message = ' ' * 75 + 'net (ms):'
            for dt in mean_dt:
                message += ' {:5.1f}'.format(dt)
            print(message)

        return logits

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        # Reshape to have size [1, C, N]
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        target = target.squeeze().unsqueeze(0)

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, target)

        # Combined loss
        return self.output_loss

    def loss_rsmix(self, outputs, labels, labels_b, lam):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        B = int(outputs.shape[0])
        loss = 0
        for i in range(B):

            loss_a = self.loss(outputs[i].unsqueeze(0), labels[i].unsqueeze(0))
            loss_b = self.loss(outputs[i].unsqueeze(0), labels_b[i].unsqueeze(0))
            loss += loss_a * (1-lam[i]) + loss_b * lam[i]

        self.output_loss = loss/B

        return self.output_loss

    def accuracy(self, outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total












