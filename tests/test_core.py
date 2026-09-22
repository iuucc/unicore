from __future__ import annotations

import os
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

from unicore_eeg.model import ARTIFACT_NAMES, SparseRouter, UniCOREEGConfig
from unicore_eeg.on004784 import task_to_labels
from unicore_eeg.physiomotion import map_annotation
from unicore_eeg.synthetic import SyntheticEEGDataset


class RouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = SparseRouter(UniCOREEGConfig())

    @staticmethod
    def tokens(probabilities: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, experts = probabilities.shape
        return {
            "probabilities": probabilities,
            "severities": torch.ones(batch, experts),
            "priorities": torch.zeros(batch, experts),
        }

    def test_clean_bypass_has_zero_route(self) -> None:
        outputs = self.router(self.tokens(torch.full((2, 6), 0.1)))
        self.assertTrue(outputs["bypass"].all())
        self.assertEqual(float(outputs["route"].sum()), 0.0)

    def test_oracle_uses_only_labeled_experts(self) -> None:
        labels = torch.tensor([[1, 0, 1, 0, 0, 0]], dtype=torch.float32)
        outputs = self.router(self.tokens(torch.full((1, 6), 0.5)), mode="oracle", oracle_labels=labels)
        torch.testing.assert_close(outputs["route"], torch.tensor([[0.5, 0.0, 0.5, 0.0, 0.0, 0.0]]))

    def test_disabled_expert_is_never_routed(self) -> None:
        disabled = torch.tensor([[True, False, False, False, False, False]])
        outputs = self.router(self.tokens(torch.tensor([[0.99, 0.8, 0.7, 0.6, 0.5, 0.4]])), disabled_experts=disabled)
        self.assertEqual(float(outputs["route"][0, 0]), 0.0)


class DatasetTests(unittest.TestCase):
    def test_six_expert_record_shapes(self) -> None:
        sample = SyntheticEEGDataset(samples=1, length=256, sample_rate=128)[0]
        self.assertEqual(len(ARTIFACT_NAMES), 6)
        self.assertEqual(tuple(sample["artifacts"].shape), (6, 1, 256))
        self.assertEqual(tuple(sample["labels"].shape), (6,))

    def test_heldout_condition_targets_unknown(self) -> None:
        dataset = SyntheticEEGDataset(samples=12, length=256, sample_rate=128, mode="first_experiment")
        sample = dataset[7]
        self.assertTrue(sample["disabled_experts"][0])
        self.assertEqual(float(sample["label_mask"][0]), 0.0)
        self.assertEqual(float(sample["labels"][-1]), 1.0)

    def test_physiomotion_combination_mapping(self) -> None:
        labels, family = map_annotation("blink_hor_headm")
        self.assertEqual(float(labels[1]), 1.0)
        self.assertEqual(float(labels[4]), 1.0)
        self.assertEqual(family, 4)

    def test_on004784_task_mapping(self) -> None:
        clean, clean_family = task_to_labels("Brain")
        all_labels, all_family = task_to_labels("All")
        self.assertEqual(float(clean.sum()), 0.0)
        self.assertEqual(clean_family, 0)
        self.assertEqual(float(all_labels[1]), 1.0)
        self.assertEqual(float(all_labels[2]), 1.0)
        self.assertEqual(float(all_labels[4]), 1.0)
        self.assertEqual(all_family, 5)


if __name__ == "__main__":
    unittest.main()
