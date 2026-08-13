import inspect
import tempfile
import unittest

import torch
import transformers
from packaging.version import Version

from sensenova_u1.models.neo_unify.transformers_compat import (
    causal_mask_kwargs,
    pretrained_dtype_kwargs,
)


class TransformersCompatibilityTest(unittest.TestCase):
    def assert_inference_paths(self, model) -> None:
        model.eval()
        input_ids = torch.tensor([[1, 3, 4]])

        with torch.no_grad():
            prefill = model(input_ids=input_ids, use_cache=True)
            decode = model(
                input_ids=torch.tensor([[5]]),
                past_key_values=prefill.past_key_values,
                use_cache=True,
            )
            generated = model.generate(input_ids, max_new_tokens=2, do_sample=False)

        self.assertEqual(prefill.logits.shape, (1, 3, 32))
        self.assertEqual(decode.logits.shape, (1, 1, 32))
        self.assertEqual(decode.past_key_values.get_seq_length(), 4)
        self.assertEqual(generated.shape[0], 1)
        self.assertGreaterEqual(generated.shape[1], input_ids.shape[1] + 1)
        self.assertLessEqual(generated.shape[1], input_ids.shape[1] + 2)

    def test_dense_model_forward_cache_decode_and_generate(self) -> None:
        from sensenova_u1.models.neo_unify.configuration_neo_chat import NEOLLMConfig
        from sensenova_u1.models.neo_unify.modeling_qwen3 import Qwen3ForCausalLM

        config = NEOLLMConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config)
        self.assert_inference_paths(model)

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            model.save_pretrained(checkpoint_dir)
            reloaded = Qwen3ForCausalLM.from_pretrained(
                checkpoint_dir,
                config=config,
                **pretrained_dtype_kwargs(torch.float32),
            )

        self.assertEqual(reloaded.dtype, torch.float32)

    def test_auto_config_and_model_registration(self) -> None:
        from transformers import AutoConfig, AutoModel

        from sensenova_u1.models.neo_unify import NEOChatConfig, NEOChatModel

        config = AutoConfig.for_model("neo_chat")

        self.assertIsInstance(config, NEOChatConfig)
        self.assertIs(AutoModel._model_mapping[NEOChatConfig], NEOChatModel)

    def test_composite_model_initializes_tied_weight_metadata(self) -> None:
        from sensenova_u1.models.neo_unify import NEOChatConfig, NEOChatModel
        from sensenova_u1.models.neo_unify.modeling_neo_vit import precompute_rope_freqs_sincos

        config = NEOChatConfig(
            vision_config={
                "architectures": ["NEOVisionModel"],
                "num_channels": 3,
                "patch_size": 2,
                "hidden_size": 8,
                "llm_hidden_size": 16,
                "downsample_ratio": 0.5,
                "max_position_embeddings_vision": 16,
            },
            llm_config={
                "architectures": ["Qwen3ForCausalLM"],
                "vocab_size": 32,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "max_position_embeddings": 64,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": 0,
            },
            downsample_ratio=0.5,
            template="neo1_0",
            fm_head_layers=2,
            use_pixel_head=False,
            concat_time_token_num=0,
            noise_scale=1.0,
            noise_scale_mode="fixed",
            noise_scale_base_image_seq_len=64,
            add_noise_scale_embedding=False,
            noise_scale_max_value=8.0,
            time_schedule="standard",
            time_shift_type="exponential",
            base_shift=0.5,
            max_shift=1.15,
            base_image_seq_len=64,
            max_image_seq_len=4096,
        )
        model = NEOChatModel(config)

        if Version(transformers.__version__).major >= 5:
            self.assertTrue(hasattr(model, "all_tied_weights_keys"))
            self.assertIsInstance(model.all_tied_weights_keys, dict)

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            model.save_pretrained(checkpoint_dir)
            reloaded = NEOChatModel.from_pretrained(
                checkpoint_dir,
                config=config,
                **pretrained_dtype_kwargs(torch.float32),
            )

        pixel_values = torch.randn(16, 12)
        grid_hw = torch.tensor([[4, 4]])
        vision_output = reloaded.vision_model(pixel_values=pixel_values, grid_hw=grid_hw)
        expected_cos, expected_sin = precompute_rope_freqs_sincos(4, 16)
        embeddings = reloaded.vision_model.embeddings

        self.assertTrue(torch.isfinite(vision_output.last_hidden_state).all())
        torch.testing.assert_close(embeddings.cos_cached_x, expected_cos)
        torch.testing.assert_close(embeddings.sin_cached_x, expected_sin)
        torch.testing.assert_close(embeddings.cos_cached_y, expected_cos)
        torch.testing.assert_close(embeddings.sin_cached_y, expected_sin)

    def test_moe_model_forward_cache_decode_and_generate(self) -> None:
        from sensenova_u1.models.neo_unify.configuration_neo_chat import NEOMoELLMConfig
        from sensenova_u1.models.neo_unify.modeling_qwen3_moe import Qwen3MoeForCausalLM

        config = NEOMoELLMConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            num_experts=2,
            num_experts_per_tok=1,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        config._attn_implementation = "eager"
        self.assert_inference_paths(Qwen3MoeForCausalLM(config))

    def test_spatial_rope_theta_is_preserved_for_non_default_rope(self) -> None:
        from sensenova_u1.models.neo_unify.configuration_neo_chat import NEOLLMConfig
        from sensenova_u1.models.neo_unify.modeling_qwen3 import Qwen3Attention

        config = NEOLLMConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            rope_theta_hw=20000.0,
            rope_scaling={"rope_type": "linear", "factor": 2.0},
        )
        spatial_config = Qwen3Attention(config, layer_idx=0).rotary_emb_hw.config

        self.assertEqual(spatial_config.rope_theta, 20000.0)
        if isinstance(getattr(spatial_config, "rope_parameters", None), dict):
            self.assertEqual(spatial_config.rope_parameters["rope_theta"], 20000.0)

    def test_causal_mask_kwargs_follow_installed_transformers_signature(self) -> None:
        from transformers.masking_utils import create_causal_mask

        parameters = inspect.signature(create_causal_mask).parameters
        inputs_embeds = torch.zeros(1, 2, 4)
        cache_position = torch.arange(2)

        kwargs = causal_mask_kwargs(
            create_causal_mask,
            config=object(),
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=cache_position.unsqueeze(0),
        )

        embedding_name = "inputs_embeds" if "inputs_embeds" in parameters else "input_embeds"
        self.assertIs(kwargs[embedding_name], inputs_embeds)
        self.assertEqual("cache_position" in kwargs, "cache_position" in parameters)
        self.assertTrue(set(kwargs) <= set(parameters))

    def test_pretrained_dtype_keyword_uses_public_installed_signature(self) -> None:
        kwargs = pretrained_dtype_kwargs(torch.float32)
        self.assertEqual(kwargs, {"dtype": torch.float32})


if __name__ == "__main__":
    unittest.main()
