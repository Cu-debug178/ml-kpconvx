import os
import sys
import threading
import unittest

import numpy as np
import torch
from easydict import EasyDict


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tasks.validation import (
    _record_validation_cycle_sample,
    full_cloud_segmentation_confusion,
    get_validation_mode,
)


def _scene_dataset_class():
    pyramid_module = sys.modules.get('utils.torch_pyramid')
    if pyramid_module is not None and not hasattr(pyramid_module, 'build_full_pyramid'):
        del sys.modules['utils.torch_pyramid']
    from data_handlers.scene_seg import SceneSegDataset
    return SceneSegDataset


class _ProjectionDataset:

    pred_values = np.array([0, 1], dtype=np.int32)
    val_labels = [np.array([0, 1, 1], dtype=np.int32)]
    input_labels = [np.array([0, 1], dtype=np.int32)]
    test_proj = [np.array([0, 1, 1], dtype=np.int32)]

    @staticmethod
    def probs_to_preds(probs):
        return np.argmax(probs, axis=1).astype(np.int32)


class _MultiWorkerMetadataDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.data_sampler = 'regular'
        self.set = 'validation'
        self.worker_lock = torch.multiprocessing.Lock()
        self.reg_sampling_i = torch.zeros(1, dtype=torch.long).share_memory_()
        self.reg_votes = torch.zeros(1, dtype=torch.long).share_memory_()
        self.reg_sampling_size = torch.tensor([2], dtype=torch.long).share_memory_()
        self.reg_sample_pts = torch.tensor(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32
        ).share_memory_()
        self.reg_sample_clouds = torch.tensor([3, 4], dtype=torch.long).share_memory_()

    def __len__(self):
        return 6

    def __getitem__(self, index):
        _, _, metadata = self.sample_input_center(return_sampling_meta=True)
        return torch.tensor(metadata, dtype=torch.long)

    def get_reg_sampling_size(self):
        return int(self.reg_sampling_size.item())

    def new_reg_sampling_pts(self):
        # In-place update mirrors the cycle-validation shared queue path.
        self.reg_sample_pts.copy_(torch.flip(self.reg_sample_pts, dims=[0]))
        self.reg_sample_clouds.copy_(torch.flip(self.reg_sample_clouds, dims=[0]))


class ValidationCycleTests(unittest.TestCase):

    def make_val_data(self):
        return EasyDict(
            proportions=np.array([2.0, 2.0], dtype=np.float32),
            cycle_states={},
            completed_cycle_ids=set(),
        )

    def test_cycle_waits_for_all_indices_and_skips_duplicates(self):
        val_data = self.make_val_data()
        completed = []
        class_zero = np.array([[2, 0], [0, 0]], dtype=np.int64)
        class_one = np.array([[0, 0], [0, 2]], dtype=np.int64)

        # Index 1 arrives first, and the duplicate must not be counted twice.
        _record_validation_cycle_sample(
            (4, 1, 2, class_one), val_data, None, 101, completed
        )
        _record_validation_cycle_sample(
            (4, 1, 2, class_one), val_data, None, 101, completed
        )
        self.assertEqual(completed, [])
        np.testing.assert_array_equal(
            val_data.cycle_states['4']['confusion'], class_one
        )

        _record_validation_cycle_sample(
            (4, 0, 2, class_zero), val_data, None, 102, completed
        )
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0]['vote_id'], 4)
        self.assertEqual(completed[0]['start_epoch'], 101)
        self.assertEqual(completed[0]['end_epoch'], 102)
        self.assertEqual(completed[0]['sample_count'], 2)
        self.assertAlmostEqual(completed[0]['miou'], 100.0, places=3)

    def test_interleaved_votes_use_independent_accumulators(self):
        val_data = self.make_val_data()
        completed = []
        conf = np.eye(2, dtype=np.int64)

        _record_validation_cycle_sample((7, 0, 2, conf), val_data, None, 20, completed)
        _record_validation_cycle_sample((8, 0, 2, conf), val_data, None, 20, completed)
        _record_validation_cycle_sample((7, 1, 2, conf), val_data, None, 21, completed)

        self.assertEqual([cycle['vote_id'] for cycle in completed], [7])
        self.assertEqual(val_data.cycle_states['8']['seen_indices'], {0})

    def test_full_cloud_confusion_uses_reprojection(self):
        probs = [np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)]
        confusion = full_cloud_segmentation_confusion(_ProjectionDataset(), probs)
        np.testing.assert_array_equal(
            confusion, np.array([[1, 0], [0, 2]], dtype=np.int64)
        )

    def test_full_identity_is_not_the_default(self):
        cfg = EasyDict(
            train=EasyDict(validation_mode='partial'),
            data=EasyDict(task='cloud_segmentation'),
        )
        self.assertEqual(get_validation_mode(cfg), 'partial')

    def test_regular_sampling_metadata_tracks_vote_boundaries(self):
        SceneSegDataset = _scene_dataset_class()
        dataset = SceneSegDataset.__new__(SceneSegDataset)
        dataset.data_sampler = 'regular'
        dataset.set = 'validation'
        dataset.worker_lock = threading.Lock()
        dataset.reg_sampling_i = torch.zeros(1, dtype=torch.long)
        dataset.reg_votes = torch.zeros(1, dtype=torch.long)
        dataset.reg_sample_pts = torch.tensor(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32
        )
        dataset.reg_sample_clouds = torch.tensor([3, 4], dtype=torch.long)
        dataset.new_reg_sampling_pts = lambda: None

        first = dataset.sample_input_center(return_sampling_meta=True)
        second = dataset.sample_input_center(return_sampling_meta=True)
        third = dataset.sample_input_center(return_sampling_meta=True)

        self.assertEqual((first[0], first[2]), (3, (0, 0, 2)))
        self.assertEqual((second[0], second[2]), (4, (0, 1, 2)))
        self.assertEqual((third[0], third[2]), (3, (1, 0, 2)))

    def test_new_vote_records_new_queue_size(self):
        SceneSegDataset = _scene_dataset_class()
        dataset = SceneSegDataset.__new__(SceneSegDataset)
        dataset.data_sampler = 'regular'
        dataset.set = 'validation'
        dataset.worker_lock = threading.Lock()
        dataset.reg_sampling_i = torch.zeros(1, dtype=torch.long)
        dataset.reg_votes = torch.zeros(1, dtype=torch.long)
        dataset.reg_sample_pts = torch.tensor(
            [[1.0, 0.0, 0.0]], dtype=torch.float32
        )
        dataset.reg_sample_clouds = torch.tensor([3], dtype=torch.long)

        def make_new_queue():
            dataset.reg_sample_pts = torch.tensor(
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32
            )
            dataset.reg_sample_clouds = torch.tensor([3, 4], dtype=torch.long)

        dataset.new_reg_sampling_pts = make_new_queue
        dataset.sample_input_center(return_sampling_meta=True)
        next_sample = dataset.sample_input_center(return_sampling_meta=True)
        self.assertEqual(next_sample[2], (1, 0, 2))

    def test_multi_worker_metadata_covers_each_vote_once(self):
        SceneSegDataset = _scene_dataset_class()
        _MultiWorkerMetadataDataset.sample_input_center = (
            SceneSegDataset.sample_input_center
        )
        loader = torch.utils.data.DataLoader(
            _MultiWorkerMetadataDataset(),
            batch_size=1,
            num_workers=2,
            persistent_workers=False,
        )

        by_vote = {}
        for metadata in loader:
            vote_id, sampling_index, sampling_size = metadata[0].tolist()
            self.assertEqual(sampling_size, 2)
            by_vote.setdefault(vote_id, set()).add(sampling_index)

        self.assertEqual(by_vote, {0: {0, 1}, 1: {0, 1}, 2: {0, 1}})


if __name__ == '__main__':
    unittest.main()
