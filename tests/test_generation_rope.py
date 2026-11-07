import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch

from sensenova_u1.models.neo_unify import modeling_qwen3, modeling_qwen3_moe
from sensenova_u1.models.neo_unify.configuration_neo_chat import NEOLLMConfig, NEOMoELLMConfig
from sensenova_u1.utils.layer_offload import LayerOffloadWrapper


class GenerationRopeTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_rope_waits_for_wrapped_first_layer_prefetch(self) -> None:
        config = NEOLLMConfig(
            vocab_size=32,
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=64,
        )
        config._attn_implementation = "eager"
        model = modeling_qwen3.Qwen3Model(config).eval().to(device="cuda", dtype=torch.bfloat16)
        kwargs = dict(
            inputs_embeds=torch.randn(1, 4, 64, device="cuda", dtype=torch.bfloat16),
            indexes=torch.tensor([[5, 5, 5, 5], [0, 0, 1, 1], [0, 1, 0, 1]], device="cuda"),
            image_gen_indicators=torch.ones(1, 4, dtype=torch.bool, device="cuda"),
            attention_mask={"full_attention": None},
            use_cache=False,
        )
        with torch.inference_mode():
            expected = model(**kwargs).last_hidden_state.clone()
            model.cpu()
            wrapper = LayerOffloadWrapper(model, "layers", torch.device("cuda"), prefetch_count=2)
            prefetch = wrapper._prefetcher.prefetch
            rotary = model.layers[0].self_attn.rotary_emb
            compute_rope = rotary.forward

            def delayed_prefetch(idx, groups):
                if idx == 0 and not wrapper._store.is_on_gpu(idx, groups):
                    with torch.cuda.stream(wrapper._prefetcher._stream):
                        torch.cuda._sleep(300_000_000)
                prefetch(idx, groups)

            def checked_rope(*args):
                # A pending event has not yet been waited on by the compute
                # stream, even if the DMA happened to finish in a fast run.
                self.assertNotIn(0, wrapper._prefetcher._events)
                return compute_rope(*args)

            try:
                with (
                    patch.object(wrapper._prefetcher, "prefetch", side_effect=delayed_prefetch),
                    patch.object(rotary, "forward", side_effect=checked_rope),
                ):
                    for _ in range(3):
                        actual = wrapper(**kwargs).last_hidden_state
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            finally:
                wrapper.teardown()

    def test_generation_reuses_static_rope_without_changing_outputs(self) -> None:
        for config_type, model_type, module in (
            (NEOLLMConfig, modeling_qwen3.Qwen3Model, modeling_qwen3),
            (NEOMoELLMConfig, modeling_qwen3_moe.Qwen3MoeModel, modeling_qwen3_moe),
        ):
            for rope_scaling, training in (
                (None, False),
                (None, True),
                ({"rope_type": "linear", "factor": 2.0}, False),
            ):
                with self.subTest(model=model_type.__name__, rope_scaling=rope_scaling, training=training):
                    config = config_type(
                        vocab_size=32,
                        hidden_size=32,
                        intermediate_size=48,
                        num_hidden_layers=2,
                        num_attention_heads=2,
                        num_key_value_heads=2,
                        head_dim=16,
                        max_position_embeddings=64,
                        num_experts=2,
                        num_experts_per_tok=1,
                        moe_intermediate_size=16,
                        rope_scaling=rope_scaling,
                    )
                    config._attn_implementation = "eager"
                    model = model_type(config).train(training)
                    x = torch.randn(2, 4, 32)
                    indexes = torch.tensor([[5, 5, 5, 5], [0, 0, 1, 1], [0, 1, 0, 1]])
                    kwargs = dict(
                        inputs_embeds=x,
                        indexes=indexes,
                        image_gen_indicators=torch.ones(2, 4, dtype=torch.bool),
                        attention_mask={"full_attention": None},
                        use_cache=False,
                    )
                    with torch.inference_mode():
                        for _ in range(2):
                            with patch.object(module, "_generation_rope_cache", return_value=None):
                                expected = model(**kwargs).last_hidden_state
                            with ExitStack() as stack:
                                calls = []
                                for layer in model.layers:
                                    for rope in (layer.self_attn.rotary_emb, layer.self_attn.rotary_emb_hw):
                                        calls.append(
                                            stack.enter_context(patch.object(rope, "forward", wraps=rope.forward))
                                        )
                                actual = model(**kwargs).last_hidden_state
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            self.assertEqual(
                                sum(call.call_count for call in calls),
                                3 if rope_scaling is None and not training else 6,
                            )
                            # Reusing or mutating the position tensor between
                            # requests must never reuse stale trigonometry.
                            indexes[0] += 1


if __name__ == "__main__":
    unittest.main()
