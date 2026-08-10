#!/usr/bin/env python3
"""Run one deterministic single-view Identity S3DIS evaluation."""

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
    ComposeAugment,
    FloorCentering,
    HeightNormalize,
    RandomDrop,
    RandomFullColor,
    RandomJitter,
    RandomScaleFlip,
)

SEED = 57106803


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def identity_transform(cfg):
    a_cfg = cfg.augment_test
    transforms = [
        RandomDrop(p=-1.0, fps=False),
        RandomScaleFlip(
            scale=[1.0, 1.0], anisotropic=False, flip_p=[0.0, 0.0, 0.0]
        ),
        RandomJitter(sigma=0.0, clip=0.0),
        FloorCentering(),
    ]
    if getattr(a_cfg, "chromatic_norm", False):
        transforms.append(
            ChromaticNormalize(
                color_mean=[0.5136457, 0.49523646, 0.44921124],
                color_std=[0.18308958, 0.18415008, 0.19252081],
            )
        )
    transforms.append(RandomFullColor(p=0.0))
    if getattr(a_cfg, "height_norm", False):
        transforms.append(HeightNormalize())
    return ComposeAugment(transforms)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--weight_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    args = parser.parse_args()

    seed_everything(SEED)
    log_path = os.path.abspath(args.log_path)
    weight_path = os.path.abspath(args.weight_path)
    cfg = load_cfg(log_path)
    cfg.data.path = os.path.abspath(args.dataset_path)
    cfg.test.batch_limit = 1
    cfg.test.in_radius = 100.0
    cfg.test.max_steps_per_epoch = 9999999
    cfg.test.max_votes = 1
    cfg.test.test_momentum = 0.0
    cfg.test.fixed_sampling = True
    cfg.augment_test.anisotropic = False
    cfg.augment_test.scale = [1.0, 1.0]
    cfg.augment_test.flips = [0.0, 0.0, 0.0]
    cfg.augment_test.jitter = 0.0
    cfg.augment_test.color_drop = 0.0
    cfg.augment_test.chromatic_contrast = False
    cfg.augment_test.chromatic_all = False
    cfg.augment_test.pts_drop_p = -1.0

    original_dataset = standalone_test.S3DIRDataset

    class IdentityDataset(original_dataset):
        def __init__(self, *dataset_args, **dataset_kwargs):
            super().__init__(*dataset_args, **dataset_kwargs)
            self.augmentation_transform = identity_transform(cfg)

    standalone_test.S3DIRDataset = IdentityDataset
    try:
        print("protocol: identity-single-view", flush=True)
        print(f"seed: {SEED}", flush=True)
        print(f"weight: {weight_path}", flush=True)
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
