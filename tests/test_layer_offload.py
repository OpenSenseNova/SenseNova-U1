import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch import nn

from sensenova_u1.models.neo_unify.modeling_neo_chat import NEOChatModel
from sensenova_u1.utils.checkpoint_loading import add_offload_args
from sensenova_u1.utils.layer_offload import (
    _ALL_TENSOR_GROUPS,
    _GROUP_GENERATION,
    _GROUP_SHARED,
    _GROUP_UNDERSTANDING,
    LayerOffloadWrapper,
    _LayerStore,
    _partition_layer_tensor_names,
    _PrefixWeightStore,
    _required_tensor_groups,
    _resident_memory_limit,
    _resolve_modules,
    _unique_module_parameters,
)
from sensenova_u1.utils.offload import (
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


class LayerOffloadPartitionTest(unittest.TestCase):
    def test_layer_store_reports_parameter_identity_as_managed(self) -> None:
        parameter = Mock()
        pinned = Mock()
        parameter.data.pin_memory.return_value = pinned
        layer = Mock()
        layer.named_parameters.return_value = (("weight", parameter),)
        layer.named_buffers.return_value = ()

        store = _LayerStore([layer], torch.device("cuda"))

        self.assertEqual(store.managed_tensor_ids(), {id(parameter)})

    def test_prefix_only_modules_deduplicate_tied_parameters(self) -> None:
        embedding = nn.Embedding(8, 4)
        lm_head = nn.Linear(4, 8, bias=False)
        lm_head.weight = embedding.weight

        parameters = _unique_module_parameters((embedding, lm_head))

        self.assertEqual(parameters, (embedding.weight,))

    def test_wrapper_switches_prefix_only_weights_with_inference_phase(self) -> None:
        wrapper = object.__new__(LayerOffloadWrapper)
        nn.Module.__init__(wrapper)
        wrapper._prefix_weight_store = Mock()
        wrapper._accel = Mock()
        wrapper._target_device = torch.device("cuda")

        LayerOffloadWrapper.set_inference_phase(wrapper, "prefix")
        wrapper._prefix_weight_store.move_to_target.assert_called_once_with()

        LayerOffloadWrapper.set_inference_phase(wrapper, "denoise")
        wrapper._accel.synchronize.assert_called_once_with(device=wrapper._target_device)
        wrapper._prefix_weight_store.evict_to_cpu.assert_called_once_with()

        with self.assertRaisesRegex(ValueError, "Unsupported inference phase"):
            LayerOffloadWrapper.set_inference_phase(wrapper, "decode")

        wrapper._prefix_weight_store = None
        with self.assertRaisesRegex(ValueError, "Unsupported inference phase"):
            LayerOffloadWrapper.set_inference_phase(wrapper, "decode")

    def test_prefix_weight_store_reuses_pinned_cpu_backing(self) -> None:
        source = Mock()
        pinned = Mock()
        target = Mock()
        source.pin_memory.return_value = pinned
        pinned.to.return_value = target
        parameter = Mock()
        parameter.data = source
        module = Mock()
        module.parameters.return_value = (parameter,)

        store = _PrefixWeightStore((module,), torch.device("cuda"))
        self.assertIs(parameter.data, pinned)

        store.move_to_target()
        pinned.to.assert_called_once_with(torch.device("cuda"), non_blocking=True)
        self.assertIs(parameter.data, target)

        store.evict_to_cpu()
        self.assertIs(parameter.data, pinned)

    def test_prefix_weight_store_does_not_partially_replace_parameters_when_pinning_fails(self) -> None:
        first_source = Mock()
        first_source.pin_memory.return_value = Mock()
        second_source = Mock()
        second_source.pin_memory.side_effect = RuntimeError("pin failed")
        first_parameter = Mock(data=first_source)
        second_parameter = Mock(data=second_source)
        module = Mock()
        module.parameters.return_value = (first_parameter, second_parameter)

        with self.assertRaisesRegex(RuntimeError, "pin failed"):
            _PrefixWeightStore((module,), torch.device("cuda"))

        self.assertIs(first_parameter.data, first_source)
        self.assertIs(second_parameter.data, second_source)

    def test_model_forwards_inference_phase_to_optional_offload_callback(self) -> None:
        callback = Mock()
        model = SimpleNamespace(_layer_offload_phase_callback=callback)

        NEOChatModel._notify_layer_offload_phase(model, "prefix")
        NEOChatModel._notify_layer_offload_phase(model, "denoise")

        self.assertEqual([call.args for call in callback.call_args_list], [("prefix",), ("denoise",)])
        NEOChatModel._notify_layer_offload_phase(SimpleNamespace(), "denoise")

    def test_resolves_model_declared_prefix_only_modules(self) -> None:
        model = nn.Module()
        model.language_model = nn.Module()
        model.language_model.model = nn.Module()
        model.language_model.model.embed_tokens = nn.Embedding(8, 4)
        model.language_model.lm_head = nn.Linear(4, 8, bias=False)

        modules = _resolve_modules(model, NEOChatModel._denoise_offload_module_paths)

        self.assertEqual(
            modules,
            (model.language_model.model.embed_tokens, model.language_model.lm_head),
        )

    def test_wrapper_restores_existing_phase_callback(self) -> None:
        previous_callback = Mock()
        model = nn.Module()
        model._layer_offload_phase_callback = previous_callback
        wrapper = object.__new__(LayerOffloadWrapper)
        nn.Module.__init__(wrapper)
        wrapper._model = model

        LayerOffloadWrapper._install_phase_callback(wrapper)
        self.assertEqual(model._layer_offload_phase_callback, wrapper.set_inference_phase)

        LayerOffloadWrapper._restore_phase_callback(wrapper)
        self.assertIs(model._layer_offload_phase_callback, previous_callback)

    def test_wrapper_removes_installed_phase_callback_when_model_had_none(self) -> None:
        model = nn.Module()
        wrapper = object.__new__(LayerOffloadWrapper)
        nn.Module.__init__(wrapper)
        wrapper._model = model

        LayerOffloadWrapper._install_phase_callback(wrapper)
        LayerOffloadWrapper._restore_phase_callback(wrapper)

        self.assertNotIn("_layer_offload_phase_callback", model.__dict__)

    def test_wrapper_does_not_remove_callback_it_did_not_install(self) -> None:
        previous_callback = Mock()
        model = nn.Module()
        model._layer_offload_phase_callback = previous_callback
        wrapper = object.__new__(LayerOffloadWrapper)
        nn.Module.__init__(wrapper)
        wrapper._model = model
        wrapper._phase_callback_installed = False

        LayerOffloadWrapper._restore_phase_callback(wrapper)

        self.assertIs(model._layer_offload_phase_callback, previous_callback)

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

    def test_offload_cli_exposes_fast_mode_budget_controls(self) -> None:
        parser = argparse.ArgumentParser()
        add_offload_args(parser)

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
