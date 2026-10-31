import unittest
from unittest import mock

import torch
from packaging.version import Version

from sensenova_u1.models.neo_unify import modeling_qwen3_moe
from sensenova_u1.models.neo_unify.configuration_neo_chat import NEOMoELLMConfig
from sensenova_u1.models.neo_unify.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
from sensenova_u1.utils.layer_offload import _release_stale_runtime_weight_caches


def _config(*, top_k: int = 2, norm_topk_prob: bool = True) -> NEOMoELLMConfig:
    return NEOMoELLMConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=3,
        num_experts_per_tok=top_k,
        norm_topk_prob=norm_topk_prob,
    )


def _set_experts_implementation(config: NEOMoELLMConfig, implementation: str) -> None:
    config.experts_implementation = implementation


def _reference_grouped_mm(
    input_: torch.Tensor,
    weight: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    output = input_.new_empty(input_.shape[0], weight.shape[-1])
    start = 0
    for expert_idx, end in enumerate(offsets.tolist()):
        output[start:end] = input_[start:end] @ weight[expert_idx]
        start = end
    return output


class GroupedGemmExpertsTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_packing_preserves_weights_from_prefetch_stream(self) -> None:
        config = _config()
        config.hidden_size = 128
        config.moe_intermediate_size = 256
        block = Qwen3MoeSparseMoeBlock(config).eval()
        expected = {name: weight.detach().clone() for name, weight in block.experts.named_parameters()}
        prefetch_stream = torch.cuda.Stream()
        with torch.cuda.stream(prefetch_stream):
            block.cuda()
        compute_stream = torch.cuda.current_stream()
        compute_stream.wait_stream(prefetch_stream)

        # Keep copies pending while the prefetch stream tries to reuse the
        # original allocations, as it can when offload prefetches another layer.
        torch.cuda._sleep(500_000_000)
        self.assertTrue(block.experts._pack_weights())
        with torch.cuda.stream(prefetch_stream):
            replacements = [torch.empty_like(weight, device="cuda").fill_(float("nan")) for weight in expected.values()]
        torch.cuda.synchronize()
        for name, weight in block.experts.named_parameters():
            torch.testing.assert_close(weight.cpu(), expected[name], rtol=0, atol=0)
        del replacements

    def test_custom_projection_replacement_releases_stale_packing(self) -> None:
        class CustomLinear(torch.nn.Linear):
            pass

        block = Qwen3MoeSparseMoeBlock(_config()).eval()
        self.assertTrue(block.experts._pack_weights())
        block.experts[1].up_proj = CustomLinear(16, 16, bias=False)
        self.assertFalse(block.experts._pack_weights())
        self.assertIsNone(block.experts._packed_gate_up_proj)
        self.assertIsNone(block.experts._packed_down_proj)

    def test_expert_slices_preserve_modulelist_behavior(self) -> None:
        experts = Qwen3MoeSparseMoeBlock(_config()).experts
        for indices in (slice(None, 2), slice(None, None, -1), slice(0, 0)):
            sliced = experts[indices]
            self.assertIsInstance(sliced, torch.nn.ModuleList)
            self.assertEqual(list(sliced), list(experts)[indices])
        self.assertIs(experts[-1], list(experts)[-1])

    def test_pytorch_28_cuda_architecture_guard(self) -> None:
        supported = modeling_qwen3_moe._cuda_grouped_mm_arch_supported
        self.assertFalse(supported(Version("2.8.0"), (8, 0)))
        self.assertTrue(supported(Version("2.8.0"), (9, 0)))
        self.assertFalse(supported(Version("2.8.0"), (10, 0)))
        self.assertFalse(supported(Version("2.9.0"), (7, 0)))
        self.assertTrue(supported(Version("2.9.0"), (8, 0)))
        self.assertTrue(supported(Version("2.9.0"), (10, 0)))

    def test_grouped_mm_stride_alignment_guard(self) -> None:
        supported = modeling_qwen3_moe._grouped_mm_strides_supported
        aligned_input = torch.empty(2, 4)
        aligned_weight = torch.empty(3, 8, 4).transpose(1, 2)
        self.assertTrue(supported(aligned_input))
        self.assertTrue(supported(aligned_weight))

        unaligned_input = torch.empty(2, 3)
        unaligned_weight = torch.empty(3, 8, 3).transpose(1, 2)
        self.assertFalse(supported(unaligned_input))
        self.assertFalse(supported(unaligned_weight))

    def test_cpu_float64_runtime_is_rejected(self) -> None:
        self.assertFalse(
            modeling_qwen3_moe._grouped_mm_runtime_supported(
                torch.device("cpu"),
                torch.float64,
            )
        )

    def test_grouped_matches_eager_for_topk_variants(self) -> None:
        for top_k, norm_topk_prob in ((1, True), (2, False), (3, True)):
            with self.subTest(top_k=top_k, norm_topk_prob=norm_topk_prob):
                torch.manual_seed(100 + top_k)
                config = _config(top_k=top_k, norm_topk_prob=norm_topk_prob)
                block = Qwen3MoeSparseMoeBlock(config).eval()
                hidden_states = torch.randn(2, 5, config.hidden_size)

                _set_experts_implementation(config, "eager")
                with torch.no_grad():
                    eager_output = block(hidden_states)

                _set_experts_implementation(config, "grouped_mm")
                with (
                    mock.patch.object(
                        modeling_qwen3_moe,
                        "_grouped_mm_runtime_supported",
                        return_value=True,
                    ),
                    mock.patch.object(modeling_qwen3_moe, "_grouped_mm_supported", return_value=True),
                    mock.patch.object(
                        modeling_qwen3_moe,
                        "_grouped_mm",
                        side_effect=_reference_grouped_mm,
                    ) as grouped_mm,
                    torch.no_grad(),
                ):
                    grouped_output = block(hidden_states)

                self.assertEqual(grouped_mm.call_count, 2)
                self.assertEqual(block.experts._last_experts_implementation, "grouped_mm")
                torch.testing.assert_close(grouped_output, eager_output, rtol=1e-5, atol=1e-6)

    def test_packing_preserves_checkpoint_keys_and_parameter_identity(self) -> None:
        config = _config()
        block = Qwen3MoeSparseMoeBlock(config).eval()
        keys_before = tuple(block.state_dict())
        parameters_before = tuple(block.parameters())

        _set_experts_implementation(config, "grouped_mm")
        with (
            mock.patch.object(
                modeling_qwen3_moe,
                "_grouped_mm_runtime_supported",
                return_value=True,
            ),
            mock.patch.object(modeling_qwen3_moe, "_grouped_mm_supported", return_value=True),
            mock.patch.object(modeling_qwen3_moe, "_grouped_mm", side_effect=_reference_grouped_mm),
            torch.no_grad(),
        ):
            block(torch.randn(2, 4, config.hidden_size))

        self.assertTrue(block.experts._weights_are_packed())
        self.assertEqual(tuple(block.state_dict()), keys_before)
        self.assertEqual(tuple(block.parameters()), parameters_before)
        self.assertFalse(any("packed" in key for key in block.state_dict()))

        gate_up = block.experts._packed_gate_up_proj
        down = block.experts._packed_down_proj
        for expert_idx, expert in enumerate(block.experts):
            self.assertEqual(
                expert.gate_proj.weight.untyped_storage().data_ptr(),
                gate_up.untyped_storage().data_ptr(),
            )
            self.assertEqual(
                expert.up_proj.weight.untyped_storage().data_ptr(),
                gate_up.untyped_storage().data_ptr(),
            )
            self.assertEqual(
                expert.down_proj.weight.untyped_storage().data_ptr(),
                down.untyped_storage().data_ptr(),
            )
            self.assertEqual(
                expert.gate_proj.weight.storage_offset(),
                expert_idx * 2 * config.moe_intermediate_size * config.hidden_size,
            )

        # The project merges LoRA deltas in-place.  Parameter views must update
        # the packed backing storage immediately rather than leave a stale copy.
        with torch.no_grad():
            block.experts[0].gate_proj.weight[0, 0].add_(1)
        self.assertEqual(
            block.experts._packed_gate_up_proj[0, 0, 0],
            block.experts[0].gate_proj.weight[0, 0],
        )

    def test_training_uses_eager_and_preserves_expert_gradients(self) -> None:
        config = _config()
        _set_experts_implementation(config, "grouped_mm")
        block = Qwen3MoeSparseMoeBlock(config).train()
        hidden_states = torch.randn(2, 5, config.hidden_size, requires_grad=True)

        with mock.patch.object(modeling_qwen3_moe, "_grouped_mm") as grouped_mm:
            output = block(hidden_states)
            output.square().mean().backward()

        grouped_mm.assert_not_called()
        self.assertEqual(block.experts._last_experts_implementation, "eager")
        self.assertIsNotNone(hidden_states.grad)
        self.assertIsNotNone(block.gate.weight.grad)
        self.assertTrue(any(expert.gate_proj.weight.grad is not None for expert in block.experts))

    def test_frozen_experts_with_activation_grad_still_use_eager(self) -> None:
        config = _config()
        _set_experts_implementation(config, "grouped_mm")
        block = Qwen3MoeSparseMoeBlock(config).eval().requires_grad_(False)
        hidden_states = torch.randn(2, 5, config.hidden_size, requires_grad=True)

        with mock.patch.object(modeling_qwen3_moe, "_grouped_mm") as grouped_mm:
            block(hidden_states).square().mean().backward()

        grouped_mm.assert_not_called()
        self.assertEqual(block.experts._last_experts_implementation, "eager")
        self.assertIsNotNone(hidden_states.grad)

    def test_grouped_preflights_both_projection_shapes(self) -> None:
        config = _config()
        _set_experts_implementation(config, "grouped_mm")
        block = Qwen3MoeSparseMoeBlock(config).eval()
        hidden_states = torch.randn(2, 5, config.hidden_size)

        with (
            mock.patch.object(
                modeling_qwen3_moe,
                "_grouped_mm_runtime_supported",
                return_value=True,
            ),
            mock.patch.object(
                modeling_qwen3_moe,
                "_grouped_mm_supported",
                side_effect=(True, False),
            ) as supported,
            mock.patch.object(modeling_qwen3_moe, "_grouped_mm") as grouped_mm,
            torch.no_grad(),
        ):
            output = block(hidden_states)

        self.assertEqual(supported.call_count, 2)
        grouped_mm.assert_not_called()
        self.assertEqual(block.experts._last_experts_implementation, "eager")
        self.assertEqual(output.shape, hidden_states.shape)

    def test_unsupported_grouped_runtime_does_not_pack_weights(self) -> None:
        config = _config()
        _set_experts_implementation(config, "grouped_mm")
        block = Qwen3MoeSparseMoeBlock(config).eval()

        with (
            mock.patch.object(
                modeling_qwen3_moe,
                "_grouped_mm_runtime_supported",
                return_value=False,
            ),
            mock.patch.object(block.experts, "_pack_weights") as pack_weights,
            torch.no_grad(),
        ):
            block(torch.randn(2, 4, config.hidden_size))

        pack_weights.assert_not_called()
        self.assertEqual(block.experts._last_experts_implementation, "eager")

    def test_stale_packed_storage_is_released_after_parameter_rebind(self) -> None:
        config = _config()
        block = Qwen3MoeSparseMoeBlock(config).eval()
        self.assertTrue(block.experts._pack_weights())
        old_packed = block.experts._packed_gate_up_proj

        block.experts[0].gate_proj.weight.data = block.experts[0].gate_proj.weight.data.clone()
        _release_stale_runtime_weight_caches(block)

        self.assertIsNone(block.experts._packed_gate_up_proj)
        self.assertIsNone(block.experts._packed_down_proj)
        self.assertNotEqual(
            block.experts[0].gate_proj.weight.untyped_storage().data_ptr(),
            old_packed.untyped_storage().data_ptr(),
        )

    def test_custom_linear_wrapper_forces_eager_fallback(self) -> None:
        class WrappedLinear(torch.nn.Linear):
            def forward(self, input_: torch.Tensor) -> torch.Tensor:
                self.was_called = True
                return super().forward(input_)

        config = _config()
        _set_experts_implementation(config, "grouped_mm")
        block = Qwen3MoeSparseMoeBlock(config).eval()
        self.assertTrue(block.experts._pack_weights())

        original = block.experts[0].gate_proj
        wrapped = WrappedLinear(config.hidden_size, config.moe_intermediate_size, bias=False)
        wrapped.weight = original.weight
        wrapped.was_called = False
        block.experts[0].gate_proj = wrapped

        with (
            mock.patch.object(
                modeling_qwen3_moe,
                "_grouped_mm_runtime_supported",
                return_value=True,
            ),
            torch.no_grad(),
        ):
            block(torch.randn(2, 4, config.hidden_size))

        self.assertTrue(wrapped.was_called)
        self.assertEqual(block.experts._last_experts_implementation, "eager")
        self.assertIsNone(block.experts._packed_gate_up_proj)
        self.assertIsNone(block.experts._packed_down_proj)

    def test_module_apply_preserves_packed_storage(self) -> None:
        config = _config()
        _set_experts_implementation(config, "grouped_mm")
        block = Qwen3MoeSparseMoeBlock(config).eval()
        self.assertTrue(block.experts._pack_weights())

        block.to(dtype=torch.float64)
        self.assertTrue(block.experts._weights_are_packed())
        self.assertTrue(all(parameter.dtype == torch.float64 for parameter in block.parameters()))

        with (
            mock.patch.object(
                modeling_qwen3_moe,
                "_grouped_mm_runtime_supported",
                return_value=True,
            ),
            mock.patch.object(modeling_qwen3_moe, "_grouped_mm_supported", return_value=True),
            mock.patch.object(modeling_qwen3_moe, "_grouped_mm", side_effect=_reference_grouped_mm),
            torch.no_grad(),
        ):
            block(torch.randn(2, 4, config.hidden_size, dtype=torch.float64))

        self.assertEqual(block.experts._packed_gate_up_proj.dtype, torch.float64)

    def test_empty_token_input_retains_reference_behavior(self) -> None:
        config = _config()
        _set_experts_implementation(config, "grouped_mm")
        block = Qwen3MoeSparseMoeBlock(config).eval()

        with torch.no_grad():
            output = block(torch.empty(2, 0, config.hidden_size))

        self.assertEqual(output.shape, (2, 0, config.hidden_size))
        self.assertEqual(block.experts._last_experts_implementation, "eager")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_real_cuda_grouped_mm_matches_eager(self) -> None:
        if not modeling_qwen3_moe._grouped_mm_runtime_supported(torch.device("cuda"), torch.bfloat16):
            self.skipTest("installed PyTorch/GPU does not support BF16 grouped MM")

        config = _config()
        block = Qwen3MoeSparseMoeBlock(config).eval().to(device="cuda", dtype=torch.bfloat16)
        hidden_states = torch.randn(
            2,
            5,
            config.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )

        _set_experts_implementation(config, "eager")
        with torch.no_grad():
            eager_output = block(hidden_states)
        _set_experts_implementation(config, "grouped_mm")
        with torch.no_grad():
            grouped_output = block(hidden_states)

        self.assertEqual(block.experts._last_experts_implementation, "grouped_mm")
        torch.testing.assert_close(grouped_output, eager_output, rtol=2e-2, atol=2e-3)

    def test_default_is_eager_and_does_not_prepack(self) -> None:
        config = _config()
        block = Qwen3MoeSparseMoeBlock(config).eval()
        with mock.patch.object(block.experts, "_pack_weights") as pack, torch.no_grad():
            block(torch.randn(2, 4, config.hidden_size))
            self.assertEqual(modeling_qwen3_moe.prepare_moe_experts_for_inference(block), 0)
        pack.assert_not_called()
        self.assertEqual(config.experts_implementation, "eager")

    def test_training_no_grad_does_not_pack_or_use_grouped(self) -> None:
        config = _config()
        config.experts_implementation = "grouped_mm"
        block = Qwen3MoeSparseMoeBlock(config).train()
        with mock.patch.object(block.experts, "_pack_weights") as pack, torch.no_grad():
            block(torch.randn(2, 4, config.hidden_size))
            self.assertEqual(modeling_qwen3_moe.prepare_moe_experts_for_inference(block), 0)
        pack.assert_not_called()
        self.assertEqual(block.experts._last_experts_implementation, "eager")

    def test_autocast_keeps_linear_dtype_policy(self) -> None:
        config = _config()
        config.experts_implementation = "grouped_mm"
        block = Qwen3MoeSparseMoeBlock(config).eval()
        with (
            mock.patch.object(block.experts, "_grouped_forward") as grouped,
            torch.no_grad(),
            torch.autocast("cpu", dtype=torch.bfloat16),
        ):
            actual = block(torch.randn(2, 4, config.hidden_size))
        grouped.assert_not_called()
        self.assertEqual(actual.dtype, torch.float32)

    def test_forward_hooks_and_custom_expert_forwards_are_not_bypassed(self) -> None:
        for target in ("expert", "projection", "activation", "custom_forward"):
            with self.subTest(target=target):
                config = _config(top_k=3)
                block = Qwen3MoeSparseMoeBlock(config).eval()
                self.assertTrue(block.experts._pack_weights())
                expert = block.experts[0]
                if target == "custom_forward":
                    original = expert.forward
                    expert.forward = lambda x: original(x) + 1
                else:
                    child = {"expert": expert, "projection": expert.gate_proj, "activation": expert.act_fn}[target]
                    child.register_forward_hook(lambda module, args, output: output + 1)
                hidden = torch.randn(2, 4, config.hidden_size)
                with torch.no_grad():
                    expected = block(hidden)
                config.experts_implementation = "grouped_mm"
                with (
                    mock.patch.object(modeling_qwen3_moe, "_grouped_mm_runtime_supported", return_value=True),
                    mock.patch.object(modeling_qwen3_moe, "_grouped_mm") as grouped,
                    torch.no_grad(),
                ):
                    actual = block(hidden)
                grouped.assert_not_called()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_inference_packing_then_training_matches_eager_gradients(self) -> None:
        import copy

        config = _config(top_k=3)
        config.experts_implementation = "grouped_mm"
        block = Qwen3MoeSparseMoeBlock(config).eval()
        reference = copy.deepcopy(block)
        reference.experts.config.experts_implementation = "eager"
        optimizer = torch.optim.SGD(block.parameters(), lr=0.01)
        with torch.inference_mode():
            self.assertTrue(block.experts._pack_weights())
        self.assertFalse(block.experts._packed_gate_up_proj.is_inference())
        block.train()
        reference.train()
        x = torch.randn(2, 5, config.hidden_size, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        block(x).square().sum().backward()
        reference(ref_x).square().sum().backward()
        torch.testing.assert_close(x.grad, ref_x.grad, rtol=0, atol=0)
        for actual, expected in zip(block.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
        optimizer.step()
        torch.optim.SGD(reference.parameters(), lr=0.01).step()
        for actual, expected in zip(block.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_reentrant_checkpoint_matches_eager_training(self) -> None:
        import copy

        from torch.utils.checkpoint import checkpoint

        config = _config(top_k=3)
        config.experts_implementation = "grouped_mm"
        block = Qwen3MoeSparseMoeBlock(config).train()
        reference = copy.deepcopy(block)
        reference.experts.config.experts_implementation = "eager"
        x = torch.randn(2, 5, config.hidden_size, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        with mock.patch.object(block.experts, "_grouped_forward") as grouped:
            actual = checkpoint(block, x, use_reentrant=True)
            actual.square().sum().backward()
        expected = reference(ref_x)
        expected.square().sum().backward()
        grouped.assert_not_called()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual_p, expected_p in zip(block.parameters(), reference.parameters()):
            torch.testing.assert_close(actual_p.grad, expected_p.grad, rtol=0, atol=0)

    def test_packed_module_can_move_to_meta(self) -> None:
        block = Qwen3MoeSparseMoeBlock(_config()).eval()
        self.assertTrue(block.experts._pack_weights())
        block.to("meta")
        self.assertTrue(all(p.device.type == "meta" for p in block.parameters()))
        self.assertIsNone(block.experts._packed_gate_up_proj)

    def test_assign_load_invalidates_packed_weights(self) -> None:
        block = Qwen3MoeSparseMoeBlock(_config()).eval()
        state = {key: value.clone() + 1 for key, value in block.state_dict().items()}
        self.assertTrue(block.experts._pack_weights())
        block.load_state_dict(state, assign=True)
        self.assertIsNone(block.experts._packed_gate_up_proj)
        self.assertTrue(block.experts._pack_weights())
        for key, value in block.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)

    def test_real_cpu_grouped_mm_with_empty_experts(self) -> None:
        if not modeling_qwen3_moe._grouped_mm_runtime_supported(torch.device("cpu"), torch.float32):
            self.skipTest("installed PyTorch has no CPU grouped MM")
        block = Qwen3MoeSparseMoeBlock(_config()).eval()
        hidden = torch.randn(7, 16)
        selected = torch.tensor([[0, 2]]).expand(7, -1)
        weights = torch.tensor([[0.75, 0.25]]).expand(7, -1)
        with torch.no_grad():
            expected = block.experts(hidden, selected, weights)
        # The block's expert collection holds the shared config.
        block.experts.config.experts_implementation = "grouped_mm"
        with torch.no_grad():
            actual = block.experts(hidden, selected, weights)
        self.assertEqual(block.experts._last_experts_implementation, "grouped_mm")
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
