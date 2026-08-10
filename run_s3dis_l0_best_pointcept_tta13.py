"""Evaluate the Standalone L0 best checkpoint with Pointcept's fixed 13-TTA.

Pointcept's evaluator cannot load a Standalone ``model_state_dict`` tar file.
This runner therefore reuses the Standalone model/test path and applies the
same fixed geometry set (rot4-scale3-xflip-v2) at the dataset boundary.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch


ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.environ.get(
    "KP_CONVX_PROJECT_DIR", os.path.join(ROOT, "Standalone", "KPConvX")
)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from experiments.S3DIS import test_S3DIS as standalone_test  # noqa: E402
from utils.config import load_cfg  # noqa: E402
from utils.transform import (  # noqa: E402
    ChromaticNormalize,
    FloorCentering,
    HeightNormalize,
    RandomDrop,
    RandomFullColor,
)


SEED = 57106803
TTA13_SIGNATURE = "rot4-scale3-xflip-v2"


class FixedPointceptTTA13:
    """Apply Pointcept's exact 13 transforms, adapted to Standalone inputs.

    The Standalone loader already selects a crop and keeps source indices, so
    predictions from every transform are accumulated onto the same cloud
    points.  Its trained preprocessing (floor centering and color/height
    normalization) is retained; only the random test geometry is replaced by
    Pointcept's deterministic geometry sequence.
    """

    def __init__(self, cfg, dataset):
        self.dataset = dataset
        a_cfg = cfg.augment_test
        self.pre = [
            RandomDrop(p=a_cfg.pts_drop_p, fps=a_cfg.pts_drop_reg),
            FloorCentering(),
        ]
        self.post = []
        if a_cfg.chromatic_norm:
            self.post.append(
                ChromaticNormalize(
                    color_mean=[0.5136457, 0.49523646, 0.44921124],
                    color_std=[0.18308958, 0.18415008, 0.19252081],
                )
            )
        self.post.append(RandomFullColor(p=a_cfg.color_drop))
        if getattr(a_cfg, "height_norm", False):
            self.post.append(HeightNormalize())

    @staticmethod
    def _rotate_z(coord, angle_half_turns):
        theta = float(angle_half_turns) * np.pi
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        return np.dot(coord, rot.T)

    def __call__(self, coord, feat, label):
        for transform in self.pre:
            coord, feat, label = transform(coord, feat, label)

        vote = int(self.dataset.reg_votes.item())
        if not 0 <= vote < 13:
            raise RuntimeError(f"unexpected Pointcept TTA vote index: {vote}")
        if vote < 12:
            angle = (0, 0.5, 1.0, 1.5)[vote % 4]
            scale = (1.0, 0.95, 1.05)[vote // 4]
            # Pointcept's tta13_augmentations order is rotate, then scale.
            coord = self._rotate_z(coord, angle)
            coord *= np.float32(scale)
        else:
            coord[:, 0] *= -1.0

        for transform in self.post:
            coord, feat, label = transform(coord, feat, label)
        return coord, feat, label


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_path",
        default=os.path.join(
            PROJECT_DIR,
            "results",
            "s3dis_litept_l0_b12a2_seed57106803",
        ),
    )
    parser.add_argument(
        "--weight_path",
        default=None,
        help="Defaults to checkpoints/best_val_miou.tar under --log_path.",
    )
    parser.add_argument(
        "--dataset_path",
        default=os.environ.get("DATASET_PATH"),
        help="S3DIS root; may also be supplied through DATASET_PATH.",
    )
    args = parser.parse_args()

    if not args.dataset_path:
        parser.error("--dataset_path or DATASET_PATH is required")

    log_path = os.path.abspath(args.log_path)
    weight_path = args.weight_path or os.path.join(
        log_path, "checkpoints", "best_val_miou.tar"
    )
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(weight_path)
    if not os.path.isdir(log_path):
        raise FileNotFoundError(log_path)

    seed_everything(SEED)
    cfg = load_cfg(log_path)
    cfg.data.path = args.dataset_path
    cfg.test.batch_limit = 1
    cfg.test.in_radius = 100.0
    cfg.test.max_steps_per_epoch = 9999999
    cfg.test.max_votes = 13
    # Pointcept's fixed TTA sums the 13 predictions for the same fragments;
    # it does not use the Standalone legacy EMA vote smoothing or resample
    # crop centers between augmentations.
    cfg.test.test_momentum = 0.0
    cfg.test.fixed_sampling = True
    cfg.augment_test.anisotropic = False
    cfg.augment_test.jitter = 0
    cfg.augment_test.color_drop = 0.0
    cfg.augment_test.chromatic_contrast = False
    cfg.augment_test.chromatic_all = False
    cfg.augment_test.pts_drop_p = -1.0

    original_dataset = standalone_test.S3DIRDataset

    class TTA13Dataset(original_dataset):
        def __init__(self, *dataset_args, **dataset_kwargs):
            super().__init__(*dataset_args, **dataset_kwargs)
            self.augmentation_transform = FixedPointceptTTA13(cfg, self)

    standalone_test.S3DIRDataset = TTA13Dataset
    try:
        print(f"Pointcept TTA13 signature: {TTA13_SIGNATURE}")
        print(f"seed: {SEED}")
        print(f"weight: {weight_path}")
        standalone_test.test_S3DIS_log(
            log_path,
            cfg,
            weight_path=weight_path,
            save_visu=False,
            profile=False,
        )
    finally:
        standalone_test.S3DIRDataset = original_dataset
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
