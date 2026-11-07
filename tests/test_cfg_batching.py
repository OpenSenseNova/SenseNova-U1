import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sensenova_u1.models.neo_unify.cfg_batching import BatchedCFGCache, cfg_batch_has_headroom
from sensenova_u1.models.neo_unify.configuration_neo_chat import NEOLLMConfig, NEOMoELLMConfig
from sensenova_u1.models.neo_unify.modeling_neo_chat import NEOChatModel, prepare_flash_kv_cache
from sensenova_u1.models.neo_unify.modeling_qwen3 import Qwen3Model
from sensenova_u1.models.neo_unify.modeling_qwen3_moe import Qwen3MoeModel
from sensenova_u1.utils.layer_offload import LayerOffloadWrapper


class CFGBatchingTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_batched_dense_and_moe_outputs_survive_offload_and_reduce_transfers(self):
        for config_type, model_type in ((NEOLLMConfig, Qwen3Model), (NEOMoELLMConfig, Qwen3MoeModel)):
            config = config_type(
                vocab_size=32,
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=3,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=64,
                max_position_embeddings=64,
                num_experts=3,
                num_experts_per_tok=2,
                moe_intermediate_size=128,
            )
            config._attn_implementation = "eager"
            config.experts_implementation = "grouped_mm"
            model = model_type(config).eval().to(device="cuda", dtype=torch.bfloat16)
            x = torch.randn(2, 4, 128, device="cuda", dtype=torch.bfloat16)
            caches, indexes = [], []
            with torch.inference_mode():
                for length in (3, 7):
                    positions = (
                        torch.stack((torch.arange(length), torch.zeros(length), torch.zeros(length))).long().cuda()
                    )
                    cache = model(
                        inputs_embeds=torch.randn(1, length, 128, device="cuda", dtype=torch.bfloat16),
                        indexes=positions,
                        use_cache=True,
                    ).past_key_values
                    prepare_flash_kv_cache(cache, current_len=4, batch_size=1)
                    caches.append(cache)
                    indexes.append(torch.tensor([[length] * 4, [0, 0, 1, 1], [0, 1, 0, 1]], device="cuda"))
                kwargs = dict(
                    indexes=torch.stack(indexes, dim=1),
                    attention_mask={"full_attention": None},
                    past_key_values=BatchedCFGCache(caches),
                    image_gen_indicators=torch.ones(2, 4, dtype=torch.bool, device="cuda"),
                    use_cache=True,
                    update_cache=False,
                )
                expected = model(inputs_embeds=x, **kwargs).last_hidden_state.clone()
                model.cpu()
                for prefetch, resident in ((0, False), (2, False), (2, True)):
                    with self.subTest(model=model_type.__name__, prefetch=prefetch, resident=resident):
                        wrapper = LayerOffloadWrapper(
                            model,
                            "layers",
                            torch.device("cuda"),
                            prefetch_count=prefetch,
                            keep_generation_resident=resident,
                        )
                        try:
                            for _ in range(2):
                                actual = wrapper(inputs_embeds=x, **kwargs).last_hidden_state
                                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            before = wrapper._store.transfer_stats()[0]
                            for index, cache, single_x in zip(indexes, caches, x.chunk(2)):
                                wrapper(
                                    inputs_embeds=single_x,
                                    indexes=index,
                                    past_key_values=cache,
                                    attention_mask={"full_attention": None},
                                    use_cache=True,
                                    update_cache=False,
                                    image_gen_indicators=torch.ones(1, 4, dtype=torch.bool, device="cuda"),
                                )
                            serial_bytes = wrapper._store.transfer_stats()[0] - before
                            before = wrapper._store.transfer_stats()[0]
                            wrapper(inputs_embeds=x, **kwargs)
                            batched_bytes = wrapper._store.transfer_stats()[0] - before
                            if resident:
                                self.assertEqual(batched_bytes, 0)
                            else:
                                self.assertLess(batched_bytes, serial_bytes)

                            # Fail after attention has overwritten image KV
                            # slots, but before the layer's offload post-hook.
                            # Serial retry must preserve the original prefixes.
                            def predict(embeds, index, mask, cache, *_args, **_kwargs):
                                return wrapper(
                                    inputs_embeds=embeds,
                                    indexes=index,
                                    attention_mask=mask,
                                    past_key_values=cache,
                                    use_cache=True,
                                    update_cache=False,
                                    image_gen_indicators=torch.ones(embeds.shape[:2], dtype=torch.bool, device="cuda"),
                                ).last_hidden_state

                            serial = [
                                predict(x[:1], index, {"full_attention": None}, cache)
                                for index, cache in zip(indexes, caches)
                            ]
                            state = BatchedCFGCache(caches)

                            def fail_batched(_module, inputs, _output):
                                if inputs[0].shape[0] == 2:
                                    raise torch.OutOfMemoryError("test partial decoder forward")

                            handle = model.layers[1].register_forward_hook(fail_batched, prepend=True)
                            try:
                                adapter = SimpleNamespace(
                                    _t2i_predict_v=predict, config=SimpleNamespace(llm_config=config)
                                )
                                with patch(
                                    "sensenova_u1.models.neo_unify.modeling_neo_chat.cfg_batch_has_headroom",
                                    return_value=True,
                                ):
                                    retried = NEOChatModel._t2i_predict_cfg(
                                        adapter,
                                        x[:1],
                                        indexes,
                                        caches,
                                        torch.tensor(0.5, device="cuda"),
                                        x[:1],
                                        4,
                                        cfg_cache=state,
                                    )
                                self.assertFalse(state.enabled)
                                for actual, reference in zip(retried, serial):
                                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                                self.assertEqual([cache.get_seq_length() for cache in caches], [3, 7])
                            finally:
                                handle.remove()
                        finally:
                            wrapper.teardown()

    def test_memory_guard_respects_free_memory_and_fast_workspace_budget(self):
        x = SimpleNamespace(device=torch.device("cuda"), shape=(1, 256, 4096), element_size=lambda: 2)
        config = SimpleNamespace(hidden_size=4096, intermediate_size=12288)
        with patch.object(torch.cuda, "mem_get_info", return_value=(8 * 1024**3, 80 * 1024**3)):
            self.assertTrue(cfg_batch_has_headroom(x, config, 4 * 1024**3))
            self.assertFalse(cfg_batch_has_headroom(x, config, 0))
            x.shape = (8, 8192, 4096)
            self.assertFalse(cfg_batch_has_headroom(x, config, 4 * 1024**3))
        x.shape = (1, 256, 4096)
        with patch.object(torch.cuda, "mem_get_info", return_value=(1024**3, 80 * 1024**3)):
            self.assertFalse(cfg_batch_has_headroom(x, config))

    def test_oom_retries_serial_and_disables_batching_only_for_this_request(self):
        x = torch.randn(1, 4, 64)
        indexes = (torch.zeros(3, 4, dtype=torch.long), torch.ones(3, 4, dtype=torch.long))
        caches = (object(), object())
        state = SimpleNamespace(enabled=True)
        predictor = Mock(side_effect=[torch.OutOfMemoryError("test workspace"), x, x + 1, x, x + 1])
        model = SimpleNamespace(_t2i_predict_v=predictor, config=SimpleNamespace(llm_config=object()))
        with patch(
            "sensenova_u1.models.neo_unify.modeling_neo_chat.cfg_batch_has_headroom", return_value=True
        ) as guard:
            for _ in range(2):
                cond, uncond = NEOChatModel._t2i_predict_cfg(
                    model, x, indexes, caches, torch.tensor(0.5), x, 4, cfg_cache=state
                )
                torch.testing.assert_close(cond, x)
                torch.testing.assert_close(uncond, x + 1)
            guard.assert_called_once()
        self.assertFalse(state.enabled)
        self.assertEqual(predictor.call_args_list[0].args[0].shape[0], 2)
        self.assertEqual([call.args[3] for call in predictor.call_args_list[1:]], list(caches) * 2)

    def test_non_memory_errors_are_not_silently_retried(self):
        x = torch.zeros(1, 4, 64)
        model = SimpleNamespace(
            _t2i_predict_v=Mock(side_effect=RuntimeError("bad kernel")), config=SimpleNamespace(llm_config=object())
        )
        with patch("sensenova_u1.models.neo_unify.modeling_neo_chat.cfg_batch_has_headroom", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "bad kernel"):
                NEOChatModel._t2i_predict_cfg(
                    model,
                    x,
                    (torch.zeros(3, 4),) * 2,
                    (object(),) * 2,
                    torch.tensor(0.5),
                    x,
                    4,
                    cfg_cache=SimpleNamespace(enabled=True),
                )
        model._t2i_predict_v.assert_called_once()

    def test_different_prefix_lengths_and_batch_sizes_match_serial(self):
        for config_type, model_type in ((NEOLLMConfig, Qwen3Model), (NEOMoELLMConfig, Qwen3MoeModel)):
            for batch_size in (1, 2):
                with self.subTest(model=model_type.__name__, batch_size=batch_size):
                    config = config_type(
                        vocab_size=32,
                        hidden_size=64,
                        intermediate_size=96,
                        num_hidden_layers=2,
                        num_attention_heads=2,
                        num_key_value_heads=1,
                        head_dim=32,
                        max_position_embeddings=64,
                        num_experts=3,
                        num_experts_per_tok=2,
                        moe_intermediate_size=32,
                    )
                    config._attn_implementation = "eager"
                    model = model_type(config).eval()
                    caches, indexes = [], []
                    with torch.inference_mode():
                        for length in (3, 7):
                            prefix_indexes = torch.stack(
                                (torch.arange(length), torch.zeros(length), torch.zeros(length))
                            ).long()
                            cache = model(
                                inputs_embeds=torch.randn(batch_size, length, 64),
                                indexes=prefix_indexes,
                                use_cache=True,
                            ).past_key_values
                            prepare_flash_kv_cache(cache, current_len=4, batch_size=batch_size)
                            caches.append(cache)
                            indexes.append(torch.tensor([[length] * 4, [0, 0, 1, 1], [0, 1, 0, 1]]))
                        cfg_cache = BatchedCFGCache(tuple(caches))
                        combined_indexes = torch.cat(
                            [index[:, None].expand(-1, batch_size, -1) for index in indexes], dim=1
                        )
                        for _ in range(2):
                            x = torch.randn(batch_size, 4, 64)
                            kwargs = dict(attention_mask={"full_attention": None}, update_cache=False, use_cache=True)
                            expected = [
                                model(
                                    inputs_embeds=x,
                                    indexes=index,
                                    past_key_values=cache,
                                    image_gen_indicators=torch.ones(batch_size, 4, dtype=torch.bool),
                                    **kwargs,
                                ).last_hidden_state
                                for index, cache in zip(indexes, caches)
                            ]
                            actual = model(
                                inputs_embeds=torch.cat([x, x]),
                                indexes=combined_indexes,
                                past_key_values=cfg_cache,
                                image_gen_indicators=torch.ones(2 * batch_size, 4, dtype=torch.bool),
                                **kwargs,
                            ).last_hidden_state
                            torch.testing.assert_close(actual, torch.cat(expected), rtol=1e-5, atol=1e-6)
                            self.assertEqual([c.get_seq_length() for c in caches], [3, 7])


if __name__ == "__main__":
    unittest.main()
