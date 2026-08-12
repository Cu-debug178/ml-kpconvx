#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# ----------------------------------------------------------------------------------------------------------------------
#
#   Hugues THOMAS - 06/10/2023
#
#   KPConvX project: validation.py
#       > Function for our model validation
#

# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import torch
import numpy as np
from os import makedirs, listdir
from os.path import exists, join
import time
import pickle

# PLY reader
from utils.ply import read_ply, write_ply
from utils.metrics import IoU_from_confusions, fast_confusion
from utils.mixed_precision import autocast_context

from utils.printing import underline


VALIDATION_MODES = ('partial', 'full_identity')


def get_validation_mode(cfg):
    """Validate and return the configured validation protocol."""

    mode = str(getattr(cfg.train, 'validation_mode', 'partial')).strip().lower()
    if mode not in VALIDATION_MODES:
        raise ValueError(
            "train.validation_mode must be one of {}; got {!r}".format(
                VALIDATION_MODES, mode
            )
        )
    if mode == 'full_identity' and cfg.data.task != 'cloud_segmentation':
        raise ValueError(
            "train.validation_mode='full_identity' is only supported for cloud segmentation"
        )
    return mode


def full_cloud_segmentation_confusion(dataset, cloud_probs):
    """Project subsampled predictions to the original clouds and aggregate confusion."""

    full_labels = dataset.val_labels
    if len(full_labels) == 0:
        full_labels = dataset.input_labels
    if len(cloud_probs) != len(full_labels):
        raise ValueError('Validation probabilities and full-cloud labels have different lengths')

    pred_values = np.asarray(dataset.pred_values, dtype=np.int32)
    confusion = np.zeros((len(pred_values), len(pred_values)), dtype=np.int64)
    has_projections = len(dataset.test_proj) == len(cloud_probs) and len(dataset.test_proj) > 0
    for cloud_i, (sub_probs, labels) in enumerate(zip(cloud_probs, full_labels)):
        sub_preds = dataset.probs_to_preds(sub_probs)
        labels = np.asarray(labels, dtype=np.int32)
        if has_projections:
            preds = sub_preds[dataset.test_proj[cloud_i]].astype(np.int32)
        elif sub_preds.shape[0] == labels.shape[0]:
            preds = sub_preds.astype(np.int32)
        else:
            raise ValueError('Full-cloud validation requires test_proj reprojection indices')
        confusion += fast_confusion(labels, preds, pred_values).astype(np.int64)
    return confusion

# ----------------------------------------------------------------------------------------------------------------------
#
#           Validation Choice
#       \***********************/
#


def validation_epoch(epoch, net, val_loader, cfg, val_data, device, amp_settings):

    validation_mode = get_validation_mode(cfg)
    if validation_mode == 'full_identity':
        # Each epoch is an independent deterministic single-view evaluation.
        if getattr(cfg.train, 'save_best_val_cycle', False) and 'cycle_states' in val_data:
            val_loader.dataset.reg_votes += 1
        val_data.clear()
        val_loader.dataset.reg_sampling_i.zero_()

    if cfg.data.task == 'classification':
        metric = object_classification_validation(
            epoch, net, val_loader, cfg, val_data, device, amp_settings
        )
        return {'metric': metric, 'completed_cycles': []}

    elif cfg.data.task == 'part_segmentation':
        metric = object_segmentation_validation(epoch, net, val_loader, cfg, val_data, device)
        return {'metric': metric, 'completed_cycles': []}

    elif cfg.data.task == 'multi_part_segmentation':
        metric = object_segmentation_validation(epoch, net, val_loader, cfg, val_data, device)
        return {'metric': metric, 'completed_cycles': []}

    elif cfg.data.task == 'cloud_segmentation':
        return cloud_segmentation_validation(
            epoch, net, val_loader, cfg, val_data, device, amp_settings
        )

    elif cfg.data.task == 'slam_segmentation':
        metric = slam_segmentation_validation(epoch, net, val_loader, cfg, val_data, device)
        return {'metric': metric, 'completed_cycles': []}

    elif cfg.data.task == 'normals_regression':
        metric = regression_validation(epoch, net, val_loader, cfg, val_data, device)
        return {'metric': metric, 'completed_cycles': []}
    else:
        raise ValueError('No validation method implemented for this network type')

# ----------------------------------------------------------------------------------------------------------------------
#
#           Validation Functions
#       \**************************/
#


def cloud_segmentation_validation(
    epoch,
    net,
    val_loader,
    cfg,
    val_data,
    device,
    amp_settings,
    debug=False,
):
    """
    Validation method for cloud segmentation models
    """
    
    ############
    # Initialize
    ############
    
    underline('Validation epoch {:d}'.format(epoch))
    message =  '\n                                                          Timings        '
    message += '\n Steps |   Votes   | Mem usage |      Speed      |   In   Batch  Forw  End '
    message += '\n-------|-----------|-----------|-----------------|-------------------------'
    print(message)

    t0 = time.time()

    # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
    val_smooth = cfg.test.val_momentum
    softmax = torch.nn.Softmax(1)

    # Number of classes including ignored labels
    nc_tot = cfg.data.num_classes

    # Number of classes predicted by the model
    nc_model = net.num_logits

    # Initiate global prediction over validation clouds
    if 'probs' not in val_data:
        val_data.probs = [np.zeros((l.shape[0], nc_model))
                          for l in val_loader.dataset.input_labels]
        val_data.vote_probs = [np.zeros((l.shape[0], nc_model))
                               for l in val_loader.dataset.input_labels]
        val_data.proportions = np.zeros(nc_model, dtype=np.float32)
        i = 0
        for label_value in val_loader.dataset.label_values:
            if label_value not in val_loader.dataset.ignored_labels:
                val_data.proportions[i] = np.sum([np.sum(val_lbls == label_value)
                                                  for val_lbls in val_loader.dataset.val_labels])
                i += 1

    cycle_enabled = bool(getattr(cfg.train, 'save_best_val_cycle', False))
    cycle_enabled = cycle_enabled and val_loader.dataset.data_sampler == 'regular'
    completed_cycles = []
    if cycle_enabled:
        if 'cycle_states' not in val_data:
            val_data.cycle_states = {}
        if 'completed_cycle_ids' not in val_data:
            val_data.completed_cycle_ids = set()

    #####################
    # Network predictions
    #####################

    run_batch_size = 0
    predictions = []
    targets = []

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(1)


    t1 = time.time()

    # Start validation loop
    for step, batch in enumerate(val_loader):

        # New time
        t = t[-1:]
        if 'cuda' in device.type:
            torch.cuda.synchronize(device)
        t += [time.time()]

        if 'cuda' in device.type:
            batch.to(device)

        # Update effective batch size
        mean_f = max(0.02, 1.0 / (step + 1))
        run_batch_size *= 1 - mean_f
        run_batch_size += mean_f * len(batch.in_dict.lengths0)

        if 'cuda' in device.type:
            torch.cuda.synchronize(device)
        t += [time.time()]

        # Forward pass
        with autocast_context(amp_settings, device):
            outputs = net(batch)
        outputs = outputs.float()
        
        if 'cuda' in device.type:
            torch.cuda.synchronize(device)
        t += [time.time()]

        # Get probs and labels
        stacked_probs = softmax(outputs).cpu().detach().numpy()
        labels = batch.in_dict.labels.cpu().numpy()
        lengths = batch.in_dict.lengths[0].cpu().numpy()
        lengths0 = batch.in_dict.lengths0.cpu().numpy()
        in_inds = batch.in_dict.input_inds.cpu().numpy()
        in_invs = batch.in_dict.input_invs.cpu().numpy()
        cloud_inds = batch.in_dict.cloud_inds.cpu().numpy()
        if cycle_enabled and hasattr(batch.in_dict, 'reg_vote_ids'):
            reg_vote_ids = batch.in_dict.reg_vote_ids.cpu().numpy()
            reg_sampling_inds = batch.in_dict.reg_sampling_inds.cpu().numpy()
            reg_sampling_sizes = batch.in_dict.reg_sampling_sizes.cpu().numpy()
        else:
            reg_vote_ids = np.full((len(lengths),), -1, dtype=np.int64)
            reg_sampling_inds = np.full((len(lengths),), -1, dtype=np.int64)
            reg_sampling_sizes = np.full((len(lengths),), -1, dtype=np.int64)

        # Get predictions and labels per instance
        # ***************************************

        i0 = 0
        j0 = 0
        for b_i, length in enumerate(lengths):

            # Get prediction
            length0 = lengths0[b_i]
            target = labels[i0:i0 + length]
            probs = stacked_probs[i0:i0 + length]
            inds = in_inds[j0:j0 + length0]
            invs = in_invs[j0:j0 + length0]
            c_i = cloud_inds[b_i]

            # Update current probs in whole cloud
            new_probs = probs[invs]
            val_data.probs[c_i][inds] = new_probs
            val_data.vote_probs[c_i][inds] *= val_smooth
            val_data.vote_probs[c_i][inds] += (1 - val_smooth) * new_probs

            # Stack all prediction for this epoch
            predictions.append(probs[invs])
            targets.append(target[invs])
            if cycle_enabled:
                sample_conf = fast_confusion(
                    target[invs],
                    val_loader.dataset.probs_to_preds(probs[invs]),
                    val_loader.dataset.pred_values,
                ).astype(np.int64)
                _record_validation_cycle_sample(
                    (
                        reg_vote_ids[b_i],
                        reg_sampling_inds[b_i],
                        reg_sampling_sizes[b_i],
                        sample_conf,
                    ),
                    val_data,
                    val_loader.dataset,
                    epoch,
                    completed_cycles,
                )
            i0 += length
            j0 += length0


        # Get CUDA memory stat to see what space is used on GPU
        if 'cuda' in device.type:
            cuda_stats = torch.cuda.memory_stats(device)
            used_GPU_MB = cuda_stats["allocated_bytes.all.peak"]
            _, tot_GPU_MB = torch.cuda.mem_get_info(device)
            # This is allocated-memory occupancy, not SM utilization.  Core
            # utilization is sampled externally (for example with nvidia-smi).
            gpu_usage = 100 * used_GPU_MB / tot_GPU_MB
            torch.cuda.reset_peak_memory_stats(device)
        else:
            gpu_usage = 0

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
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

        # Display
        if (t[-1] - last_display) > 1.0:
            last_display = t[-1]
            message = ' {:5d} | {:9.2f} | {:7.1f} % | {:7.1f} ins/sec | {:6.1f} {:5.1f} {:5.1f} {:5.1f}'
            print(message.format(step,
                                 val_loader.dataset.get_votes(),
                                 gpu_usage,
                                 run_batch_size / np.sum(mean_dt),
                                 1000 * mean_dt[0],
                                 1000 * mean_dt[1],
                                 1000 * mean_dt[2],
                                 1000 * mean_dt[3]))

    t2 = time.time()

    if get_validation_mode(cfg) == 'full_identity':
        # Select checkpoints with the metric reported on the original room points.
        sum_Confs = full_cloud_segmentation_confusion(val_loader.dataset, val_data.probs)
        t3 = time.time()
        t4 = t3
    else:
        # Keep the legacy partial-validation metric exactly as before.
        Confs = np.zeros((len(predictions), nc_model, nc_model), dtype=np.int32)
        for i, (probs, truth) in enumerate(zip(predictions, targets)):
            preds = val_loader.dataset.probs_to_preds(probs)
            Confs[i, :, :] = fast_confusion(
                truth, preds, val_loader.dataset.pred_values
            ).astype(np.int32)

        t3 = time.time()

        # Balance sampled fragments with the full validation class proportions.
        sum_Confs = np.sum(Confs, axis=0).astype(np.float32)
        sum_Confs *= np.expand_dims(
            val_data.proportions / (np.sum(sum_Confs, axis=1) + 1e-6), 1
        )

        t4 = time.time()

    # Objects IoU
    IoUs = IoU_from_confusions(sum_Confs)

    t5 = time.time()

    # Saving (optionnal)
    if cfg.exp.saving:

        # Name of saving file
        test_file = join(cfg.exp.log_dir, 'val_IoUs.txt')

        # Line to write:
        line = ''
        for IoU in IoUs:
            line += '{:.3f} '.format(IoU)
        line = line + '\n'

        # Write in file
        if exists(test_file):
            with open(test_file, "a") as text_file:
                text_file.write(line)
        else:
            with open(test_file, "w") as text_file:
                text_file.write(line)

        # # Save potentials
        # pot_path = join(cfg.exp.log_dir, 'potentials')
        # if not exists(pot_path):
        #     makedirs(pot_path)
        # files = val_loader.dataset.scene_files
        # for i, file_path in enumerate(files):
        #     pot_points = np.array(val_loader.dataset.pot_trees[i].data, copy=False)
        #     cloud_name = file_path.split('/')[-1]
        #     pot_name = join(pot_path, cloud_name)
        #     pots = val_loader.dataset.potentials[i].numpy().astype(np.float32)
        #     write_ply(pot_name,
        #                 [pot_points.astype(np.float32), pots],
        #                 ['x', 'y', 'z', 'pots'])

    t6 = time.time()

    # Print instance mean
    mIoU = 100 * np.mean(IoUs)
    print('\n{:s} mean IoU = {:.1f}%'.format(cfg.data.name, mIoU))
    print()


    # Save predicted cloud occasionally
    # *********************************

    # Create validation folder
    val_path = join(cfg.exp.log_dir, 'validation')
    if not exists(val_path):
        makedirs(val_path)
    current_votes = val_loader.dataset.get_votes()
    last_vote = int(np.floor(current_votes))

    # Check if vote has already been saved
    saved_votes = np.sort([int(l.split('_')[1]) for l in listdir(val_path) if  l.startswith('conf_')])
    if last_vote not in saved_votes:

        conf_path = join(val_path, 'conf_{:d}_{:d}.txt'.format(last_vote, epoch + 1))
        conf_vote_path = join(val_path, 'vote_conf_{:d}_{:d}.txt'.format(last_vote, epoch + 1))

        # Save the subsampled input clouds with latest predictions
        files = val_loader.dataset.scene_files
        scene_confs = np.zeros((nc_model, nc_model), dtype=np.int32)
        scene_vote_confs = np.zeros((nc_model, nc_model), dtype=np.int32)
        for c_i, file_path in enumerate(files):

            # Get subsampled points from tree structure
            points = np.array(val_loader.dataset.input_trees[c_i].data, copy=False)
            if val_loader.dataset.cylindric_input:
                points = np.hstack((points, val_loader.dataset.input_z[c_i]))

            # Get probs on our own ply points
            sub_probs = val_data.probs[c_i]
            sub_vote_probs = val_data.vote_probs[c_i]

            # Get predictions
            sub_preds = val_loader.dataset.probs_to_preds(sub_probs)
            sub_vote_preds = val_loader.dataset.probs_to_preds(sub_vote_probs)

            # Path of saved validation file
            val_name = join(val_path, val_loader.dataset.scene_names[c_i] + '.ply')

            # Save file
            labels = val_loader.dataset.input_labels[c_i]

            if getattr(cfg.test, 'save_validation_clouds', True):
                write_ply(val_name,
                          [points, sub_vote_preds, sub_preds, labels.astype(np.int32)],
                          ['x', 'y', 'z', 'vote_pre', 'last_pre', 'class'])

            # Get full groundtruth labels
            labels = val_loader.dataset.val_labels[c_i].astype(np.int32)

            # Reproject preds on the evaluations points
            preds = sub_preds[val_loader.dataset.test_proj[c_i]].astype(np.int32)
            vote_preds = sub_vote_preds[val_loader.dataset.test_proj[c_i]].astype(np.int32)

            # Confusion matrix
            pred_values = np.array(cfg.data.pred_values, dtype=np.int32)
            scene_confs += fast_confusion(labels, preds, pred_values).astype(np.int32)
            scene_vote_confs += fast_confusion(labels, vote_preds, pred_values).astype(np.int32)

        # Save confusion for future use
        np.savetxt(conf_path, scene_confs, '%12d')
        
        # Save confusion for future use
        np.savetxt(conf_vote_path, scene_vote_confs, '%12d')


        IoUs1 = IoU_from_confusions(scene_confs)
        IoUs2 = IoU_from_confusions(scene_vote_confs)
        

    # Display timings
    t7 = time.time()
    if debug:
        print('\n************************\n')
        print('Validation timings:')
        print('Init ...... {:.1f}s'.format(t1 - t0))
        print('Loop ...... {:.1f}s'.format(t2 - t1))
        print('Confs ..... {:.1f}s'.format(t3 - t2))
        print('Confs bis . {:.1f}s'.format(t4 - t3))
        print('IoU ....... {:.1f}s'.format(t5 - t4))
        print('Save1 ..... {:.1f}s'.format(t6 - t5))
        print('Save2 ..... {:.1f}s'.format(t7 - t6))
        print('\n************************\n')

    if cfg.exp.saving and completed_cycles:
        cycle_file = join(cfg.exp.log_dir, 'val_cycle_IoUs.txt')
        with open(cycle_file, 'a') as text_file:
            for cycle in completed_cycles:
                line = '{:d} {:d} {:d} {:.6f}'.format(
                    cycle['vote_id'],
                    cycle['start_epoch'],
                    cycle['end_epoch'],
                    cycle['miou'],
                )
                line += ''.join(' {:.6f}'.format(value) for value in cycle['ious'])
                text_file.write(line + '\n')

    return {
        'metric': float(mIoU),
        'ious': [float(value) for value in IoUs],
        'confusion': np.asarray(sum_Confs).tolist(),
        'completed_cycles': completed_cycles,
    }


def _record_validation_cycle_sample(
    sample,
    val_data,
    dataset,
    epoch,
    completed_cycles,
):
    """Add one regular sample to its vote accumulator and finalize complete votes."""

    vote_id, sampling_index, sampling_size, sample_conf = sample
    vote_id = int(vote_id)
    sampling_index = int(sampling_index)
    sampling_size = int(sampling_size)
    if vote_id < 0 or sampling_index < 0 or sampling_size < 1:
        return

    cycle_key = str(vote_id)
    if vote_id in val_data.completed_cycle_ids:
        print('[ValCycle] duplicate completed vote={}, skipping'.format(vote_id))
        return

    states = val_data.cycle_states
    state = states.get(cycle_key)
    if state is None:
        state = {
            'confusion': np.zeros_like(sample_conf, dtype=np.int64),
            'seen_indices': set(),
            'expected_size': sampling_size,
            'start_epoch': int(epoch),
        }
        states[cycle_key] = state
    elif state['expected_size'] != sampling_size:
        raise ValueError(
            'regular validation vote {} changed size from {} to {}'.format(
                vote_id, state['expected_size'], sampling_size
            )
        )

    if sampling_index >= state['expected_size']:
        raise ValueError(
            'regular validation vote {} has out-of-range index {} (size {})'.format(
                vote_id, sampling_index, state['expected_size']
            )
        )
    if sampling_index in state['seen_indices']:
        print(
            '[ValCycle] duplicate vote={} index={}, skipping'.format(
                vote_id, sampling_index
            )
        )
        return

    state['seen_indices'].add(sampling_index)
    state['confusion'] += sample_conf
    print(
        '[ValCycle] vote={} progress={}/{} epoch={}'.format(
            vote_id,
            len(state['seen_indices']),
            state['expected_size'],
            epoch,
        )
    )

    if len(state['seen_indices']) != state['expected_size']:
        return

    raw_cycle_conf = state['confusion'].copy()
    balanced_cycle_conf = raw_cycle_conf.astype(np.float64)
    proportions = np.asarray(val_data.proportions, dtype=np.float64)
    balanced_cycle_conf *= np.expand_dims(
        proportions / (np.sum(balanced_cycle_conf, axis=1) + 1e-6), 1
    )
    cycle_ious = IoU_from_confusions(balanced_cycle_conf)
    cycle_miou = float(100 * np.mean(cycle_ious))
    cycle_result = {
        'vote_id': vote_id,
        'miou': cycle_miou,
        'ious': (100 * np.asarray(cycle_ious)).tolist(),
        'start_epoch': int(state['start_epoch']),
        'end_epoch': int(epoch),
        'sample_count': len(state['seen_indices']),
        'raw_confusion': raw_cycle_conf,
    }
    val_data.completed_cycle_ids.add(vote_id)
    completed_cycles.append(cycle_result)
    del states[cycle_key]
    print(
        '[ValCycle] completed vote={} epochs={}-{} samples={}/{} mIoU={:.3f}'.format(
            cycle_result['vote_id'],
            cycle_result['start_epoch'],
            cycle_result['end_epoch'],
            cycle_result['sample_count'],
            state['expected_size'],
            cycle_miou,
        )
    )


def object_classification_validation(
    epoch,
    net,
    val_loader,
    cfg,
    val_data,
    device,
    amp_settings,
    debug=False,
):
    """
    Validation method for classification models
    """
    
    ############
    # Initialize
    ############
    
    underline('Validation epoch {:d}'.format(epoch))
    message =  '\n                                                          Timings        '
    message += '\n Steps |   Votes   | GPU usage |      Speed      |   In   Batch  Forw  End '
    message += '\n-------|-----------|-----------|-----------------|-------------------------'
    print(message)

    t0 = time.time()

    # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
    val_smooth = cfg.test.val_momentum
    softmax = torch.nn.Softmax(1)

    # Number of classes including ignored labels
    nc_tot = cfg.data.num_classes

    # Number of classes predicted by the model
    nc_model = net.num_logits

    # Initiate global prediction over validation clouds
    if 'probs' not in val_data:
        val_data.probs = np.zeros((val_loader.dataset.n_objects, nc_model))

    #####################
    # Network predictions
    #####################

    run_batch_size = 0
    probs = []
    targets = []
    obj_inds = []

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(1)


    t1 = time.time()

    # Start validation loop
    for step, batch in enumerate(val_loader):

        # New time
        t = t[-1:]
        if 'cuda' in device.type:
            torch.cuda.synchronize(device)
        t += [time.time()]

        if 'cuda' in device.type:
            batch.to(device)

        # Update effective batch size
        mean_f = max(0.02, 1.0 / (step + 1))
        run_batch_size *= 1 - mean_f
        run_batch_size += mean_f * len(batch.in_dict.lengths[0])

        if 'cuda' in device.type:
            torch.cuda.synchronize(device)
        t += [time.time()]

        # Forward pass
        with autocast_context(amp_settings, device):
            outputs = net(batch)
        outputs = outputs.float()
        
        if 'cuda' in device.type:
            torch.cuda.synchronize(device)
        t += [time.time()]


        # Get probs and labels
        probs += [softmax(outputs).cpu().detach().numpy()]
        targets += [batch.in_dict.labels.cpu().numpy()]
        obj_inds += [batch.in_dict.obj_inds.cpu().numpy()]

        if 'cuda' in device.type:
            # Get CUDA memory stat to see what space is used on GPU
            cuda_stats = torch.cuda.memory_stats(device)
            used_GPU_MB = cuda_stats["allocated_bytes.all.peak"]
            _, tot_GPU_MB = torch.cuda.mem_get_info(device)
            gpu_usage = 100 * used_GPU_MB / tot_GPU_MB
            torch.cuda.reset_peak_memory_stats(device)
        else:
            gpu_usage = 0

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
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

        # Display
        if (t[-1] - last_display) > 1.0:
            last_display = t[-1]
            message = ' {:5d} | {:9.2f} | {:7.1f} % | {:7.1f} ins/sec | {:6.1f} {:5.1f} {:5.1f} {:5.1f}'
            print(message.format(step,
                                 val_loader.dataset.get_votes(),
                                 gpu_usage,
                                 run_batch_size / np.sum(mean_dt),
                                 1000 * mean_dt[0],
                                 1000 * mean_dt[1],
                                 1000 * mean_dt[2],
                                 1000 * mean_dt[3]))

    t2 = time.time()

    # Stack all validation predictions
    probs = np.vstack(probs)
    targets = np.hstack(targets)
    obj_inds = np.hstack(obj_inds)

    ###################
    # Voting validation
    ###################

    val_data.probs[obj_inds] = val_smooth * val_data.probs[obj_inds] + (1-val_smooth) * probs

    ############
    # Confusions
    ############

    # Compute classification results
    C1 = fast_confusion(targets,
                        val_loader.dataset.probs_to_preds(probs),
                        val_loader.dataset.pred_values)

    # Compute votes confusion
    C2 = fast_confusion(val_loader.dataset.input_labels,
                        val_loader.dataset.probs_to_preds(val_data.probs),
                        val_loader.dataset.pred_values)


    # Saving (optionnal)
    if cfg.exp.saving:
        print("Save confusions")
        conf_list = [C1, C2]
        file_list = ['val_confs.txt', 'vote_confs.txt']
        for conf, conf_file in zip(conf_list, file_list):
            test_file = join(cfg.exp.log_dir, conf_file)
            if exists(test_file):
                with open(test_file, "a") as text_file:
                    for line in conf:
                        for value in line:
                            text_file.write('%d ' % value)
                    text_file.write('\n')
            else:
                with open(test_file, "w") as text_file:
                    for line in conf:
                        for value in line:
                            text_file.write('%d ' % value)
                    text_file.write('\n')

    val_ACC = 100 * np.sum(np.diag(C1)) / (np.sum(C1) + 1e-6)
    vote_ACC = 100 * np.sum(np.diag(C2)) / (np.sum(C2) + 1e-6)
    print('Accuracies : val = {:.1f}% / vote = {:.1f}%'.format(val_ACC, vote_ACC))

    return float(vote_ACC)


def object_segmentation_validation(epoch, net, val_loader, cfg, val_data, device, debug=False):
    return


def slam_segmentation_validation(epoch, net, val_loader, cfg, val_data, device, debug=False):
    return


def regression_validation(epoch, net, val_loader, cfg, val_data, device, debug=False):
    return














