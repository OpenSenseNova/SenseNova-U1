import argparse
import unittest

import torch
from torch import nn

from sensenova_u1.utils.checkpoint_loading import add_offload_args
from sensenova_u1.utils.layer_offload import (
    _ALL_TENSOR_GROUPS,
    _GROUP_GENERATION,
    _GROUP_SHARED,
    _GROUP_UNDERSTANDING,
    _partition_layer_tensor_names,
    _required_tensor_groups,
    _resident_memory_limit,
    _resolve_optional_module_attrs,
)
from sensenova_u1.utils.offload import (
    DEFAULT_AUXILIARY_OFFLOAD_ATTRS,
    DEFAULT_FAST_ACTIVATION_RESERVE_GIB,
    DEFAULT_FAST_VRAM_FRACTION,
    DEFAULT_FAST_VRAM_HEADROOM_GIB,
    VRAM_MODE_OPTIONS,
    vram_mode_keeps_generation_resident,
    vram_mode_to_prefetch_count,
)


class _TwoBranchLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(4, 4, bias=False)
        self.self_attn.q_proj_mot_gen = nn.Linear(4, 4, bias=False)
        self.mlp = nn.Sequential(nn.Linear(4, 8, bias=False), nn.Linear(8, 4, bias=False))
        self.mlp_mot_gen = nn.Sequential(nn.Linear(4, 8, bias=False), nn.Linear(8, 4, bias=False))
        self.register_buffer("shared_scale", torch.ones(()))


class _ModelWithOptionalModules(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.embed_tokens = nn.Embedding(8, 4)
        self.language_model.lm_head = nn.Linear(4, 8, bias=False)
        self.scalar = 1


class LayerOffloadPartitionTest(unittest.TestCase):
    def test_partitions_paired_branch_tensors_and_shared_buffers(self) -> None:
        groups = _partition_layer_tensor_names(_TwoBranchLayer())

        self.assertEqual(groups["self_attn.q_proj.weight"], _GROUP_UNDERSTANDING)
        self.assertEqual(groups["self_attn.q_proj_mot_gen.weight"], _GROUP_GENERATION)
        self.assertEqual(groups["mlp.0.weight"], _GROUP_UNDERSTANDING)
        self.assertEqual(groups["mlp_mot_gen.0.weight"], _GROUP_GENERATION)
        self.assertEqual(groups["shared_scale"], _GROUP_SHARED)

    def test_plain_transformer_layer_remains_fully_shared(self) -> None:
        layer = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4))
        groups = _partition_layer_tensor_names(layer)

        self.assertTrue(groups)
        self.assertEqual(set(groups.values()), {_GROUP_SHARED})

    def test_selects_only_the_active_branch(self) -> None:
        self.assertEqual(
            _required_tensor_groups({"exist_non_image_gen_tokens": True, "exist_image_gen_tokens": False}),
            frozenset({_GROUP_SHARED, _GROUP_UNDERSTANDING}),
        )
        self.assertEqual(
            _required_tensor_groups({"exist_non_image_gen_tokens": False, "exist_image_gen_tokens": True}),
            frozenset({_GROUP_SHARED, _GROUP_GENERATION}),
        )

    def test_unknown_forward_signature_falls_back_to_all_tensors(self) -> None:
        self.assertEqual(_required_tensor_groups({}), _ALL_TENSOR_GROUPS)
        self.assertEqual(
            _required_tensor_groups(
                {
                    "exist_non_image_gen_tokens": torch.tensor(False),
                    "exist_image_gen_tokens": torch.tensor(True),
                }
            ),
            _ALL_TENSOR_GROUPS,
        )

    def test_generation_residency_keeps_safety_headroom(self) -> None:
        gib = 1024**3

        self.assertEqual(_resident_memory_limit(24 * gib), int(21.6 * gib))
        self.assertEqual(_resident_memory_limit(16 * gib), 14 * gib)
        self.assertEqual(_resident_memory_limit(2 * gib), 0)

    def test_fast_mode_uses_prefetch_and_generation_residency(self) -> None:
        self.assertEqual(VRAM_MODE_OPTIONS, ("full", "fast", "balanced", "low"))
        self.assertEqual(vram_mode_to_prefetch_count("fast"), 2)
        self.assertTrue(vram_mode_keeps_generation_resident("fast"))

        for mode in ("full", "balanced", "low"):
            self.assertFalse(vram_mode_keeps_generation_resident(mode))

    def test_generation_residency_budget_is_configurable(self) -> None:
        gib = 1024**3

        self.assertEqual(
            _resident_memory_limit(24 * gib, memory_fraction=0.85),
            int(20.4 * gib),
        )
        self.assertEqual(
            _resident_memory_limit(24 * gib, headroom_bytes=4 * gib),
            20 * gib,
        )
        self.assertEqual(
            _resident_memory_limit(24 * gib, memory_fraction=0.5, budget_bytes=20 * gib),
            20 * gib,
        )

        with self.assertRaises(ValueError):
            _resident_memory_limit(24 * gib, memory_fraction=0)
        with self.assertRaises(ValueError):
            _resident_memory_limit(24 * gib, headroom_bytes=-1)
        with self.assertRaises(ValueError):
            _resident_memory_limit(24 * gib, budget_bytes=0)

    def test_optional_auxiliary_modules_skip_missing_paths_and_deduplicate(self) -> None:
        model = _ModelWithOptionalModules()
        modules = _resolve_optional_module_attrs(
            model,
            (
                "language_model.model.embed_tokens",
                "missing.path",
                "language_model.lm_head",
                "language_model.lm_head",
            ),
        )

        self.assertEqual(
            [path for path, _module in modules],
            ["language_model.model.embed_tokens", "language_model.lm_head"],
        )
        self.assertIs(modules[0][1], model.language_model.model.embed_tokens)
        self.assertIs(modules[1][1], model.language_model.lm_head)

        with self.assertRaises(TypeError):
            _resolve_optional_module_attrs(model, ("scalar",))

    def test_offload_cli_exposes_fast_mode_budget_controls(self) -> None:
        parser = argparse.ArgumentParser()
        add_offload_args(parser)

        self.assertEqual(
            DEFAULT_AUXILIARY_OFFLOAD_ATTRS,
            ("language_model.model.embed_tokens", "language_model.lm_head"),
        )

        defaults = parser.parse_args([])
        self.assertEqual(defaults.fast_vram_fraction, DEFAULT_FAST_VRAM_FRACTION)
        self.assertEqual(defaults.fast_vram_headroom_gib, DEFAULT_FAST_VRAM_HEADROOM_GIB)
        self.assertEqual(defaults.fast_activation_reserve_gib, DEFAULT_FAST_ACTIVATION_RESERVE_GIB)
        self.assertIsNone(defaults.fast_vram_budget_gib)

        custom = parser.parse_args(
            [
                "--fast_vram_fraction",
                "0.85",
                "--fast_vram_headroom_gib",
                "3",
                "--fast_activation_reserve_gib",
                "5",
                "--fast_vram_budget_gib",
                "20.5",
            ]
        )
        self.assertEqual(custom.fast_vram_fraction, 0.85)
        self.assertEqual(custom.fast_vram_headroom_gib, 3.0)
        self.assertEqual(custom.fast_activation_reserve_gib, 5.0)
        self.assertEqual(custom.fast_vram_budget_gib, 20.5)


if __name__ == "__main__":
    unittest.main()
