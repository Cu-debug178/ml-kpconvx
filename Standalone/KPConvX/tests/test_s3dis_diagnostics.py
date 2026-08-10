import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_s3dis_difficulties import (  # noqa: E402
    hierarchy_attributes_for_batch,
    parse_gpu_compute_processes,
    run_compare,
    run_dataset_audit,
    run_profile,
)
from utils.s3dis_diagnostics import (  # noqa: E402
    confusion_matrix,
    difficulty_masks,
    estimate_quantile_thresholds,
    fixed_radius_geometry,
    load_prediction_artifact,
    metrics_from_confusion,
    patch_neighbor_recall,
    save_prediction_artifact,
    validate_paired_artifacts,
)


class GeometryDiagnosticTests(unittest.TestCase):

    def test_fixed_radius_density_boundary_and_pca(self):
        points = np.array(
            [
                [0.00, 0.00, 0.00],
                [0.04, 0.00, 0.00],
                [0.08, 0.00, 0.00],
                [0.04, 0.04, 0.00],
                [0.04, -0.04, 0.00],
            ],
            dtype=np.float32,
        )
        labels = np.array([0, 0, 1, 0, 0], dtype=np.int64)
        attributes = fixed_radius_geometry(
            points,
            labels,
            radii=(0.05, 0.10),
            pca_radius=0.10,
            max_pca_neighbors=16,
        )
        self.assertEqual(attributes["density_count_r0050mm"][0], 1)
        self.assertEqual(attributes["density_count_r0050mm"][1], 4)
        self.assertFalse(attributes["boundary_r0050mm"][0])
        self.assertTrue(attributes["boundary_r0050mm"][1])
        self.assertAlmostEqual(attributes["boundary_fraction_r0050mm"][1], 0.25)
        self.assertTrue(np.isfinite(attributes["planarity"][1]))
        self.assertLess(attributes["curvature"][1], 1e-6)

    def test_sampled_queries_still_use_complete_room_support(self):
        points = np.array([[0.01 * index, 0.0, 0.0] for index in range(8)], dtype=np.float32)
        labels = np.zeros(8, dtype=np.int64)
        attributes = fixed_radius_geometry(
            points,
            labels,
            radii=(0.021,),
            query_indices=np.array([3]),
            pca_radius=0.04,
        )
        self.assertEqual(attributes["density_count_r0021mm"].tolist(), [4])
        self.assertEqual(
            attributes["pca_neighbor_count_used"].tolist(),
            attributes["pca_neighbor_count_full"].tolist(),
        )


class MetricDiagnosticTests(unittest.TestCase):

    def test_confusion_metrics_and_masks(self):
        labels = np.array([0, 0, 1, 1])
        predictions = np.array([0, 1, 1, 1])
        confusion = confusion_matrix(labels, predictions, 2)
        self.assertTrue(np.array_equal(confusion, np.array([[1, 1], [0, 2]])))
        metrics = metrics_from_confusion(confusion)
        self.assertAlmostEqual(metrics["OA"], 75.0)
        self.assertAlmostEqual(metrics["mIoU_all"], (0.5 + 2 / 3) * 50)

        attributes = {
            "boundary_r0100mm": np.array([False, True, False, True]),
            "density_count_r0100mm": np.array([1, 2, 3, 4]),
            "grid_stage2_mixed_cell": np.array([0, 1, 0, 1]),
            "grid_stage1_cell_entropy": np.zeros(4),
        }
        thresholds = estimate_quantile_thresholds([attributes])
        masks = difficulty_masks(attributes, thresholds, point_count=4)
        self.assertEqual(masks["boundary_r0100mm"].tolist(), [False, True, False, True])
        self.assertEqual(masks["grid_stage2_pure_cell"].tolist(), [True, False, True, False])
        self.assertIn("density_count_r0100mm__low", masks)
        self.assertIn("density_count_r0100mm__high", masks)
        self.assertFalse(np.any(masks["grid_stage1_cell_entropy__high"]))

    def test_patch_union_recovers_neighbors_split_by_individual_orders(self):
        neighbors = np.array(
            [
                [0, 1, 2, 4],
                [1, 0, 3, 4],
                [2, 0, 3, 4],
                [3, 1, 2, 4],
            ]
        )
        attributes = patch_neighbor_recall(
            neighbors,
            {
                "z": np.array([0, 0, 1, 1]),
                "z-trans": np.array([0, 1, 0, 1]),
            },
        )
        self.assertAlmostEqual(attributes["patch_neighbor_recall_z"][0], 0.5)
        self.assertAlmostEqual(attributes["patch_neighbor_recall_z-trans"][0], 0.5)
        self.assertAlmostEqual(attributes["patch_neighbor_recall_union"][0], 1.0)
        self.assertAlmostEqual(attributes["patch_cut_ratio_union"][0], 0.0)

    def test_actual_hierarchy_attributes_map_back_to_stage_zero(self):
        import torch

        stage0_points = torch.tensor(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [1.1, 0.0, 0.0]]
        )
        stage1_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        trace = {
            "points": [stage0_points, stage1_points],
            "lengths": [torch.tensor([4]), torch.tensor([2])],
            "upsamples": [torch.tensor([[0], [0], [1], [1]])],
            "labels": torch.tensor([0, 1, 1, 1]),
        }
        batch = SimpleNamespace(
            in_dict=SimpleNamespace(
                neighbors=[
                    torch.tensor([[0, 1, 4], [1, 0, 4], [2, 3, 4], [3, 2, 4]]),
                    torch.tensor([[0, 1, 2], [1, 0, 2]]),
                ]
            )
        )
        cfg = SimpleNamespace(
            model=SimpleNamespace(
                litept_enabled=True,
                litept_orders="z,z-trans",
                litept_conv_stages=1,
                litept_handover_stage=0,
                litept_patch_size=1,
                in_sub_size=0.1,
                radius_scaling=2.0,
            )
        )
        attributes = hierarchy_attributes_for_batch(batch, trace, cfg)[0]
        self.assertEqual(
            attributes["grid_stage1_mixed_cell"].tolist(), [True, True, False, False]
        )
        self.assertEqual(
            attributes["__grid_stage1_cell_local_id"].tolist(), [0, 0, 1, 1]
        )
        self.assertTrue(
            np.allclose(attributes["stage1_patch_cut_ratio_union"], np.ones(4))
        )


class ArtifactAndCliTests(unittest.TestCase):

    @staticmethod
    def _points_labels():
        points = np.array(
            [
                [0.00, 0.00, 0.00],
                [0.04, 0.00, 0.00],
                [0.08, 0.00, 0.00],
                [0.12, 0.00, 0.00],
                [0.16, 0.00, 0.00],
                [0.20, 0.00, 0.00],
            ],
            dtype=np.float32,
        )
        labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        return points, labels

    def test_artifact_roundtrip_and_pair_alignment_rejection(self):
        points, labels = self._points_labels()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "room.npz"
            save_prediction_artifact(
                path,
                "Area_5_room",
                points,
                labels,
                labels,
                attributes={"boundary_r0100mm": np.zeros(labels.size, dtype=bool)},
                metadata={"votes": 1},
            )
            artifact = load_prediction_artifact(path)
            self.assertEqual(artifact["scene_name"], "Area_5_room")
            self.assertEqual(int(artifact["metadata"]["votes"]), 1)
            shifted = dict(artifact)
            shifted["points"] = artifact["points"].copy()
            shifted["points"][0, 0] += 0.1
            with self.assertRaisesRegex(ValueError, "coordinates differ"):
                validate_paired_artifacts(artifact, shifted)

    def test_gpu_process_parser(self):
        rows = parse_gpu_compute_processes(
            "26049, python, 23418 MiB\ninvalid row\n42, worker, 512 MiB\n"
        )
        self.assertEqual(
            rows,
            [
                {"pid": 26049, "process_name": "python", "used_memory_mib": 23418},
                {"pid": 42, "process_name": "worker", "used_memory_mib": 512},
            ],
        )

    def test_dataset_profile_and_compare_outputs(self):
        points, labels = self._points_labels()
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            dataset_path = temporary / "s3dis"
            for area, room_name in ((1, "office_1"), (5, "conference_1")):
                room = dataset_path / "Area_{}".format(area) / room_name
                room.mkdir(parents=True)
                np.save(room / "coord.npy", points)
                np.save(room / "color.npy", np.zeros_like(points))
                np.save(room / "segment.npy", labels)
                if area == 5:
                    np.save(room / "instance.npy", np.array([0, 0, 1, 1, 2, 2]))

            dataset_output = temporary / "dataset_output"
            run_dataset_audit(
                SimpleNamespace(
                    dataset_path=str(dataset_path),
                    output_dir=str(dataset_output),
                    geometry_split="validation",
                    geometry_max_points_per_room=6,
                    seed=3,
                    radii=[0.05, 0.10],
                    pca_radius=0.10,
                    geometry_chunk_size=8,
                    max_pca_neighbors=16,
                )
            )
            self.assertTrue((dataset_output / "dataset_class_stats.csv").is_file())
            self.assertTrue((dataset_output / "instance_stats.csv").is_file())
            with (dataset_output / "room_geometry_summary.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                geometry_rows = list(csv.DictReader(handle))
            self.assertEqual(len(geometry_rows), 1)
            self.assertIn("boundary_r0050mm_mean", geometry_rows[0])

            baseline_dir = temporary / "baseline"
            candidate_dir = temporary / "candidate"
            attributes = {
                "boundary_r0100mm": np.array([False, True, True, False, True, False]),
                "density_count_r0100mm": np.array([1, 2, 3, 4, 5, 6]),
                "grid_stage3_mixed_cell": np.array([0, 1, 1, 0, 1, 0]),
                "stage3_patch_cut_ratio_union": np.linspace(0, 1, labels.size),
            }
            baseline_predictions = np.array([0, 1, 1, 0, 2, 1])
            candidate_predictions = np.array([0, 0, 1, 1, 2, 1])
            for directory, predictions in (
                (baseline_dir, baseline_predictions),
                (candidate_dir, candidate_predictions),
            ):
                save_prediction_artifact(
                    directory / "room.npz",
                    "Area_5_room",
                    points,
                    labels,
                    predictions,
                    attributes=attributes,
                )

            profile_output = temporary / "profile_output"
            run_profile(
                SimpleNamespace(
                    prediction_dir=str(baseline_dir),
                    output_dir=str(profile_output),
                    compute_geometry=False,
                    radii=[0.10],
                    pca_radius=0.10,
                    geometry_chunk_size=8,
                    max_pca_neighbors=16,
                    seed=4,
                )
            )
            self.assertTrue((profile_output / "subset_metrics.csv").is_file())
            self.assertTrue((profile_output / "per_class_subset_metrics.csv").is_file())
            self.assertTrue((profile_output / "confusion_matrix.csv").is_file())

            compare_output = temporary / "compare_output"
            run_compare(
                SimpleNamespace(
                    baseline_predictions=str(baseline_dir),
                    candidate_predictions=str(candidate_dir),
                    output_dir=str(compare_output),
                    baseline_name="B0",
                    candidate_name="B1",
                    compute_geometry=False,
                    radii=[0.10],
                    pca_radius=0.10,
                    geometry_chunk_size=8,
                    max_pca_neighbors=16,
                    coordinate_atol=1e-5,
                    bootstrap_repeats=20,
                    seed=5,
                )
            )
            with (compare_output / "transitions_by_subset.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            all_row = next(row for row in rows if row["subset"] == "all")
            self.assertEqual(int(all_row["fixed_by_candidate"]), 2)
            self.assertEqual(int(all_row["regressed_by_candidate"]), 0)
            self.assertTrue((compare_output / "gain_by_class.csv").is_file())
            self.assertTrue((compare_output / "gain_by_class_subset.csv").is_file())


if __name__ == "__main__":
    unittest.main()
