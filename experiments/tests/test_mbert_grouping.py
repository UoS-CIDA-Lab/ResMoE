"""Public grouping and exact-budget invariants for experiment 248."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory

import torch

spec = importlib.util.spec_from_file_location(
    "mbert_modular", Path(__file__).resolve().parents[1] / "248_mbert_modular.py")
assert spec is not None and spec.loader is not None
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class GroupingTests(unittest.TestCase):
    def test_weight_groups_preserve_directions_and_equal_sizes(self) -> None:
        weight = torch.tensor([[1., .01], [2., .02], [3., .03], [4., .04],
                               [.01, 1.], [.02, 2.], [.03, 3.], [.04, 4.]])
        original = weight.clone()
        sizes, labels = experiment.gmoe_weight_groups(weight, 2, "cpu")
        self.assertTrue(torch.equal(weight, original))
        self.assertEqual(sizes.tolist(), [4., 4.])
        self.assertEqual(len(set(labels[:4].tolist())), 1)
        self.assertEqual(len(set(labels[4:].tolist())), 1)
        self.assertNotEqual(labels[0].item(), labels[4].item())
        _, repeated = experiment.gmoe_weight_groups(weight, 2, "cpu")
        self.assertTrue(torch.equal(labels, repeated))

    def test_nondivisible_groups_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            experiment.gmoe_weight_groups(torch.ones(7, 2), 2, "cpu")

    def test_cache_reuses_exact_groups_and_rejects_changed_weights(self) -> None:
        weight = torch.tensor([[1., 0.], [2., 0.], [0., 1.], [0., 2.]])
        with TemporaryDirectory() as directory:
            cache = Path(directory) / "groups.pt"
            first = experiment.gmoe_weight_groups(weight, 2, "cpu", cache_path=cache)
            repeated = experiment.gmoe_weight_groups(weight, 2, "cpu", cache_path=cache)
            self.assertTrue(torch.equal(first[1], repeated[1]))
            with self.assertRaises(ValueError):
                experiment.gmoe_weight_groups(weight * 2, 2, "cpu", cache_path=cache)

    def test_group_and_neuron_selection_execute_identical_budgets(self) -> None:
        groups, neurons = 64, 3072
        sizes = torch.full((groups,), float(neurons // groups))
        labels = torch.arange(groups).repeat_interleave(neurons // groups)
        group_scores = torch.arange(groups).float().expand(3, -1)
        # Deliberate ties expose over-budget threshold selection.
        neuron_scores = torch.ones(3, neurons)
        for keep, expected in zip((.75, .5, .35, .25, .2, .15),
                                  (2304, 1536, 1104, 768, 624, 480)):
            budget = experiment.executed_budget(keep, neurons, groups)
            self.assertEqual(budget, expected)
            gm = experiment.keep_topB_group(group_scores, sizes, labels, budget)
            nm = experiment.keep_topB_neuron(neuron_scores, budget)
            self.assertTrue(torch.equal(gm.sum(1), torch.full((3,), budget)))
            self.assertTrue(torch.equal(nm.sum(1), gm.sum(1)))

    def test_invalid_keep_is_rejected(self) -> None:
        for keep in (0., -0.1, 1.1):
            with self.assertRaises(ValueError):
                experiment.executed_budget(keep, 3072, 64)

    def test_correction_seed_is_reproducible_without_advancing_global_rng(self) -> None:
        inputs = torch.randn(16, 4)
        targets = torch.randn(16, 3)
        before = torch.random.get_rng_state().clone()
        first = experiment.mlp_fit(inputs, targets, "cpu", steps=5, hidden=8, bs=8, seed=17)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        repeated = experiment.mlp_fit(inputs, targets, "cpu", steps=5, hidden=8, bs=8, seed=17)
        changed = experiment.mlp_fit(inputs, targets, "cpu", steps=5, hidden=8, bs=8, seed=18)
        self.assertTrue(torch.equal(first(inputs), repeated(inputs)))
        self.assertFalse(torch.equal(first(inputs), changed(inputs)))


if __name__ == "__main__":
    unittest.main()
