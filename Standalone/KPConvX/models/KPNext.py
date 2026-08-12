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
from models.ktha_blocks import (KernelOccupancySignature, pool_kernel_signature,
                                shuffle_packed_signature)
from models.glskf_blocks import GLSKF_CONTEXT_CONTROLS, GLSKF_MODES, GlskfFeedback

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
        self.litept_legacy_kpconvd_encoder = bool(
            getattr(cfg.model, 'litept_legacy_kpconvd_encoder', False)
        )
        self.ktha_mode = str(getattr(cfg.model, 'ktha_mode', 'none')).lower()
        self.ktha_enabled = self.ktha_mode != 'none'
        self.ktha_source_stage = int(
            getattr(cfg.model, 'ktha_source_stage', self.litept_conv_stages)
        )
        target_value = str(getattr(cfg.model, 'ktha_target_stages', '4'))
        self.ktha_target_stages = tuple(
            int(value.strip())
            for value in target_value.split(',')
            if value.strip()
        )
        self.ktha_relation_dim = int(getattr(cfg.model, 'ktha_relation_dim', 8))
        self.ktha_hidden_dim = int(getattr(cfg.model, 'ktha_hidden_dim', 0))
        self.ktha_shuffle_geometry = bool(
            getattr(cfg.model, 'ktha_shuffle_geometry', False)
        )
        self.ktha_train_mode = str(
            getattr(cfg.model, 'ktha_train_mode', 'joint')
        ).lower()

        # Global-to-Local Semantic Kernel Feedback (GLSKF): top-down semantic
        # modulation of a shallower stage's effective kernel weights.
        self.glskf_mode = str(getattr(cfg.model, 'glskf_mode', 'none')).lower()
        self.glskf_enabled = self.glskf_mode != 'none'
        self.glskf_refine_stage = int(getattr(cfg.model, 'glskf_refine_stage', 3))
        context_value = str(getattr(cfg.model, 'glskf_context_stages', '4,5'))
        self.glskf_context_stages = tuple(
            int(value.strip())
            for value in context_value.split(',')
            if value.strip()
        )
        self.glskf_groups = int(getattr(cfg.model, 'glskf_groups', 8))
        self.glskf_hidden_dim = int(getattr(cfg.model, 'glskf_hidden_dim', 64))
        self.glskf_matched_hidden_dim = int(
            getattr(cfg.model, 'glskf_matched_hidden_dim', 0)
        )
        self.glskf_detach_context = bool(
            getattr(cfg.model, 'glskf_detach_context', False)
        )
        self.glskf_context_control = str(
            getattr(cfg.model, 'glskf_context_control', 'none')
        ).lower()
        self.glskf_deep_residual = bool(
            getattr(cfg.model, 'glskf_deep_residual', False)
        )
        self.glskf_train_mode = str(
            getattr(cfg.model, 'glskf_train_mode', 'joint')
        ).lower()
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
        if self.ktha_enabled:
            if not self.litept_enabled:
                raise ValueError('KTHA requires litept_enabled=True')
            if not self.share_kp:
                raise ValueError('KTHA requires share_kp=True to reuse the exact KP basis')
            if self.ktha_source_stage < 1 or self.ktha_source_stage >= self.num_layers:
                raise ValueError('ktha_source_stage must precede at least one later stage')
            if self.ktha_source_stage > self.litept_conv_stages:
                raise ValueError('ktha_source_stage must be a KP convolution stage')
            if not self.ktha_target_stages:
                raise ValueError('ktha_target_stages must contain at least one stage')
            if any(
                stage <= self.ktha_source_stage or stage > self.num_layers
                for stage in self.ktha_target_stages
            ):
                raise ValueError(
                    'Every KTHA target stage must follow the source and exist in the encoder'
                )
            if any(
                stage <= self.litept_conv_stages
                and stage != self.litept_handover_stage
                for stage in self.ktha_target_stages
            ):
                raise ValueError('Every KTHA target must contain token attention')
            if self.ktha_train_mode not in {'joint', 'module_head'}:
                raise ValueError("ktha_train_mode must be 'joint' or 'module_head'")
        if self.glskf_enabled:
            if self.glskf_mode not in GLSKF_MODES:
                raise ValueError('glskf_mode must be one of {}'.format(GLSKF_MODES))
            if self.glskf_context_control not in GLSKF_CONTEXT_CONTROLS:
                raise ValueError(
                    'glskf_context_control must be one of {}'.format(GLSKF_CONTEXT_CONTROLS)
                )
            if self.ktha_enabled:
                # Both directions of the loop would confound each other: a gain
                # could come from forward geometry handover or backward semantic
                # feedback and the experiment could not tell which.
                raise ValueError('KTHA and GLSKF cannot be enabled in the same run')
            if self.task != 'cloud_segmentation':
                raise ValueError('GLSKF corrects a decoder skip, so it needs cloud_segmentation')
            if not self.share_kp:
                raise ValueError('GLSKF requires share_kp=True to reuse the refined stage neighborhood')
            if cfg.model.kp_influence == 'mlp':
                raise ValueError("GLSKF requires nearest-kernel weights, not kp_influence='mlp'")
            if self.kp_mode == 'kpconv':
                raise ValueError(
                    "GLSKF requires the KPConvD-style nearest-kernel operator; "
                    "kp_mode='kpconv' is not supported"
                )
            if not 1 <= self.glskf_refine_stage < self.num_layers:
                raise ValueError('glskf_refine_stage must be a stage whose skip reaches the decoder')
            if (
                self.litept_enabled
                and self.glskf_refine_stage > self.litept_conv_stages
                and self.glskf_refine_stage != self.litept_handover_stage
            ):
                raise ValueError('glskf_refine_stage must contain a KP convolution')
            if not self.glskf_context_stages:
                raise ValueError('glskf_context_stages must contain at least one stage')
            if any(
                stage <= self.glskf_refine_stage or stage > self.num_layers
                for stage in self.glskf_context_stages
            ):
                raise ValueError(
                    'Every GLSKF context stage must follow the refined stage and exist in the encoder'
                )
            if self.glskf_train_mode not in {'joint', 'module_head'}:
                raise ValueError("glskf_train_mode must be 'joint' or 'module_head'")

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
        self._runtime_monitoring_enabled = False

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
            use_mod=self._pool_uses_kernel_attention(use_conv))

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
                    use_mod=self._pool_uses_kernel_attention(use_conv))
                setattr(self, 'pooling_{:d}'.format(layer), pooling_i)

        self.ktha_signature = None
        if self.ktha_enabled:
            source_radius = self.first_radius * (
                self.radius_scaling ** (self.ktha_source_stage - 1)
            )
            source_sigma = self.first_sigma * (
                self.radius_scaling ** (self.ktha_source_stage - 1)
            )
            self.ktha_signature = KernelOccupancySignature(
                shell_sizes=cfg.model.shell_sizes,
                radius=source_radius,
                sigma=source_sigma,
                dimension=cfg.data.dim,
                influence_mode=cfg.model.kp_influence,
                fixed_kernel_points=cfg.model.kp_fixed,
                kernel_points=self.shared_kp[self.ktha_source_stage - 1]["k_pts"],
            )

        self.glskf = None
        if self.glskf_enabled:
            refine_l = self.glskf_refine_stage - 1
            refine_radius = self.first_radius * (
                self.radius_scaling ** refine_l
            )
            refine_sigma = self.first_sigma * (
                self.radius_scaling ** refine_l
            )
            self.glskf = GlskfFeedback(
                refine_channels=adapter_channels[refine_l],
                context_channels=[
                    adapter_channels[stage - 1] for stage in self.glskf_context_stages
                ],
                shell_sizes=cfg.model.shell_sizes,
                radius=refine_radius,
                sigma=refine_sigma,
                mode=self.glskf_mode,
                groups=self.glskf_groups,
                hidden_dim=self.glskf_hidden_dim,
                matched_hidden_dim=self.glskf_matched_hidden_dim,
                dimension=cfg.data.dim,
                influence_mode=cfg.model.kp_influence,
                fixed_kernel_points=cfg.model.kp_fixed,
                norm_type=cfg.model.norm,
                bn_momentum=cfg.model.bn_momentum,
                shared_kp_data=self.shared_kp[refine_l],
                detach_context=self.glskf_detach_context,
                deep_residual=self.glskf_deep_residual,
                context_control=self.glskf_context_control,
            )

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
        self._configure_ktha_training()
        self._configure_glskf_training()
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

    def _configure_ktha_training(self):
        """Freeze L0 while training only geometry consumers and the task head."""

        if not self.ktha_enabled or self.ktha_train_mode == 'joint':
            return
        if self.fa_enabled and self.fa_train_mode != 'joint':
            raise ValueError('FastAdapter-only and KTHA-only training cannot be combined')

        for parameter in self.parameters():
            parameter.requires_grad = False
        for name, parameter in self.named_parameters():
            if '.ktha.' in name or name.startswith('head.'):
                parameter.requires_grad = True

    def _configure_glskf_training(self):
        """Freeze L0 while training only the feedback module and the task head."""

        if not self.glskf_enabled or self.glskf_train_mode == 'joint':
            return
        if self.fa_enabled and self.fa_train_mode != 'joint':
            raise ValueError('FastAdapter-only and GLSKF-only training cannot be combined')

        for parameter in self.parameters():
            parameter.requires_grad = False
        for name, parameter in self.named_parameters():
            if name.startswith('glskf.') or name.startswith('head.'):
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
        if mode and self.ktha_enabled and self.ktha_train_mode == 'module_head':
            for child_name, child_module in self.named_children():
                if child_name != 'head':
                    child_module.eval()
            for module_name, module in self.named_modules():
                if module_name.endswith('.ktha'):
                    module.train(True)
            self.head.train(True)
        if mode and self.glskf_enabled and self.glskf_train_mode == 'module_head':
            for child_name, child_module in self.named_children():
                if child_name not in {'glskf', 'head'}:
                    child_module.eval()
            self.glskf.train(True)
            self.head.train(True)
        return self

    def _encoder_stage_kind(self, layer, original_use_conv):
        """Return the stage role without changing the configured KP operator."""

        if not self.litept_enabled:
            return 'forced_kpconvd' if original_use_conv else 'configured_conv'
        if layer == self.litept_handover_stage:
            return 'handover'
        if layer <= self.litept_conv_stages:
            if self.litept_legacy_kpconvd_encoder:
                return 'forced_kpconvd'
            return 'configured_conv'
        return 'attention'

    def _pool_uses_kernel_attention(self, original_use_conv):
        """Keep LitePT pooling consistent with the explicitly selected kp_mode."""

        if self.litept_enabled:
            if self.litept_legacy_kpconvd_encoder:
                return False
            return self.kp_mode == 'kpconvx'
        return not original_use_conv

    def get_encoder_block(self, in_C, out_C, radius, sigma, cfg, layer, block_i,
                          shared_kp_data, stage_kind, drop_path):
        """Build a stage-tailored encoder block.

        Early ``configured_conv`` stages use the operator selected by
        ``kp_mode``. Late ``attention`` stages use serialized PointROPE token
        attention. A ``handover`` stage applies the configured convolution and
        token attention sequentially.
        """

        if stage_kind in {'forced_kpconvd', 'configured_conv'}:
            return self.get_residual_block(
                in_C, out_C, radius, sigma, cfg,
                shared_kp_data=shared_kp_data,
                conv_layer=(stage_kind == 'forced_kpconvd'),
                drop_path=drop_path)

        voxel_size = max(
            float(self.subsample_size) * self.radius_scaling ** (layer - 1),
            1e-6,
        )
        order = self.litept_orders[block_i % len(self.litept_orders)]
        patch_cache = self._litept_patch_caches.setdefault(
            layer, SerializedPatchCache()
        )
        geometry_mode = (
            self.ktha_mode
            if self.ktha_enabled and layer in self.ktha_target_stages
            else 'none'
        )
        geometry_signature_dim = (
            int(np.sum(cfg.model.shell_sizes)) if geometry_mode != 'none' else 0
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
                geometry_mode=geometry_mode,
                geometry_signature_dim=geometry_signature_dim,
                geometry_relation_dim=self.ktha_relation_dim,
                geometry_hidden_dim=self.ktha_hidden_dim,
            )

        if stage_kind == 'handover':
            convolution = self.get_residual_block(
                in_C, out_C, radius, sigma, cfg,
                shared_kp_data=shared_kp_data,
                conv_layer=False,
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
                geometry_mode=geometry_mode,
                geometry_signature_dim=geometry_signature_dim,
                geometry_relation_dim=self.ktha_relation_dim,
                geometry_hidden_dim=self.ktha_hidden_dim,
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






    def set_runtime_monitoring(self, enabled=True):
        """Enable cheap summary diagnostics for the next forward only.

        Training keeps this disabled except on configured monitor steps, so the
        default path has no reductions, host synchronisations, or retained
        intermediate tensors.
        """

        self._runtime_monitoring_enabled = bool(enabled)
        if self.fast_adapter is not None:
            self.fast_adapter.set_diagnostics_mode(summary=enabled, full=False)

    def runtime_monitoring_stats(self):
        stats = {}
        if self.fast_adapter is not None:
            stats.update(self.fast_adapter.diagnostics())
        if self.glskf is not None and self.glskf.last_stats:
            stats['glskf'] = dict(self.glskf.last_stats)
        return stats

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

    def _upsample_between_stages(self, feats, batch, from_l, to_l):
        """Chain the decoder upsamplers from level ``from_l`` down to ``to_l``.

        The pyramid only stores adjacent-level correspondences, so a stage-5
        feature reaches stage 3 through two successive nearest upsamples.
        """

        for l in range(from_l - 1, to_l - 1, -1):
            upsample = getattr(self, 'upsampling_{:d}'.format(l + 1))
            if self.grid_pool:
                feats = upsample(feats, batch.in_dict.upsamples[l])
            else:
                feats = upsample(feats, batch.in_dict.upsamples[l], batch.in_dict.up_distances[l])
        return feats

    def _apply_glskf(self, batch, skip_feats, deep_feats, diagnostics=False):
        """Rewrite the refined stage skip with a top-down corrected version."""

        refine_l = self.glskf_refine_stage - 1
        context = None
        if self.glskf_mode != 'matched_mlp':
            context_parts = []
            for stage in self.glskf_context_stages:
                source_l = stage - 1
                if stage == self.num_layers:
                    stage_feats = deep_feats
                else:
                    stage_feats = skip_feats[source_l]
                context_parts.append(
                    self._upsample_between_stages(stage_feats, batch, source_l, refine_l)
                )
            context = torch.cat(context_parts, dim=1)

        self.glskf.set_diagnostics(diagnostics)
        skip_feats[refine_l] = self.glskf(
            batch.in_dict.points[refine_l],
            skip_feats[refine_l],
            context,
            batch.in_dict.neighbors[refine_l],
            lengths=batch.in_dict.lengths[refine_l],
        )
        return skip_feats

    def forward(
        self,
        batch,
        verbose=False,
        return_intermediates=False,
        capture_adapter_details=False,
    ):

        # Serialization is shared by all attention blocks in a stage, but must
        # be rebuilt for each new batch because point coordinates change.
        for patch_cache in self._litept_patch_caches.values():
            patch_cache.clear()

        if self.fast_adapter is not None:
            self.fast_adapter.set_diagnostics_mode(
                summary=self._runtime_monitoring_enabled or return_intermediates,
                full=capture_adapter_details,
            )

        trace = None
        if return_intermediates:
            trace = {
                "stages": [],
                "points": [],
                "lengths": [],
                "upsamples": [],
                "labels": None,
            }

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

        if return_intermediates:
            trace["points"] = [point.detach() for point in batch.in_dict.points]
            trace["lengths"] = [length.detach() for length in batch.in_dict.lengths]
            trace["upsamples"] = [up.detach() for up in batch.in_dict.upsamples]
            if "labels" in batch.in_dict:
                trace["labels"] = batch.in_dict.labels.detach()

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
        kernel_signature = None
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
                stage_kernel_signature = kernel_signature
                if (
                    stage_kernel_signature is not None
                    and self.ktha_enabled
                    and layer in self.ktha_target_stages
                    and self.ktha_shuffle_geometry
                ):
                    stage_kernel_signature = shuffle_packed_signature(
                        stage_kernel_signature,
                        batch.in_dict.lengths[l],
                    )
                for block in block_list:
                    if isinstance(block, (LitePointTransformerBlock, LiteHandoverBlock)):
                        block_signature = None
                        if self.ktha_enabled and layer in self.ktha_target_stages:
                            if stage_kernel_signature is None:
                                raise RuntimeError(
                                    'KTHA target reached before a kernel signature was produced'
                                )
                            block_signature = stage_kernel_signature
                        feats, upcut = block(
                            batch.in_dict.points[l],
                            batch.in_dict.points[l],
                            feats,
                            batch.in_dict.neighbors[l],
                            batch.in_dict.lengths[l],
                            upcut=upcut,
                            kernel_signature=block_signature,
                        )
                    else:
                        feats, upcut = block(batch.in_dict.points[l], batch.in_dict.points[l], feats, batch.in_dict.neighbors[l], batch.in_dict.lengths[l], upcut=upcut)

            if self.ktha_enabled and layer == self.ktha_source_stage:
                cached_geometry = self.shared_kp[l] if self.share_kp else None
                kernel_signature = self.ktha_signature(
                    batch.in_dict.points[l],
                    batch.in_dict.points[l],
                    batch.in_dict.neighbors[l],
                    cached_geometry=cached_geometry,
                )

            # Compensate geometry before skip storage and downsampling.
            pre_adapter_features = feats
            if self.fast_adapter is not None:
                feats, fa_state = self.fast_adapter.forward_layer(
                    l,
                    batch.in_dict.points[l],
                    batch.in_dict.lengths[l],
                    feats,
                    fa_state,
                )

            if return_intermediates:
                trace["stages"].append({
                    "stage": l,
                    "pre_adapter_features": pre_adapter_features.detach(),
                    "post_adapter_features": feats.detach(),
                })
                
            if layer < self.num_layers:

                # Skip features
                skip_feats.append(feats)

                # Pooling
                layer_pool = getattr(self, 'pooling_{:d}'.format(layer))
                if self.grid_pool:
                    feats = layer_pool(feats, batch.in_dict.pools[l])
                else:
                    feats = layer_pool(batch.in_dict.points[l+1], batch.in_dict.points[l], feats, batch.in_dict.pools[l])
                if kernel_signature is not None:
                    kernel_signature = pool_kernel_signature(
                        kernel_signature, batch.in_dict.pools[l]
                    )

         
        if verbose:    
            torch.cuda.synchronize(batch.device())                         
            t += [time.time()]

        if self.task == 'classification':
            
            # Global pooling
            feats = self.global_pooling(feats, batch.in_dict.lengths[-1])

            
        elif self.task == 'cloud_segmentation':

            #  ------ Top-down semantic feedback ------

            if self.glskf_enabled:
                skip_feats = self._apply_glskf(
                    batch,
                    skip_feats,
                    feats,
                    diagnostics=self._runtime_monitoring_enabled or return_intermediates,
                )
                if return_intermediates:
                    trace['glskf'] = dict(self.glskf.last_stats)

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

        if return_intermediates:
            trace["adapter"] = (
                self.fast_adapter.diagnostics()
                if self.fast_adapter is not None
                else {}
            )

        # Summary monitoring is one-shot. The training loop explicitly enables
        # it again on the next requested optimizer step.
        self._runtime_monitoring_enabled = False

        if return_intermediates:
            return logits, trace
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





