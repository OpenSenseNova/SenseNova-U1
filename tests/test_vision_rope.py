import unittest

import torch

from sensenova_u1.models.neo_unify.configuration_neo_vit import NEOVisionConfig
from sensenova_u1.models.neo_unify.modeling_neo_vit import NEOVisionEmbeddings, precompute_rope_freqs_sincos


class VisionRopeTest(unittest.TestCase):
    def test_meta_initialization_defers_cpu_tables_until_use(self):
        config = NEOVisionConfig(hidden_size=1024, llm_hidden_size=16, max_position_embeddings_vision=256)
        with torch.device("meta"):
            embeddings = NEOVisionEmbeddings(config)
        self.assertIsNone(embeddings.cos_cached_x)
        expected_cos, expected_sin = precompute_rope_freqs_sincos(512, 256, device="cpu")
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda", torch.cuda.current_device()))
        for device in devices:
            with self.subTest(device=str(device)):
                embeddings._ensure_rope_cache(device)
                self.assertEqual(embeddings.cos_cached_x.device, device)
                self.assertEqual(embeddings.cos_cached_x.dtype, torch.float32)
                for name, expected in [
                    ("cos_cached_x", expected_cos),
                    ("cos_cached_y", expected_cos),
                    ("sin_cached_x", expected_sin),
                    ("sin_cached_y", expected_sin),
                ]:
                    self.assertTrue(torch.equal(getattr(embeddings, name).cpu(), expected))
                cached = embeddings.cos_cached_x
                embeddings._ensure_rope_cache(device)
                self.assertIs(embeddings.cos_cached_x, cached)
        self.assertNotIn("cos_cached_x", embeddings.state_dict())


if __name__ == "__main__":
    unittest.main()
