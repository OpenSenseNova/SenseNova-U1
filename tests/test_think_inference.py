import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sensenova_u1.models.neo_unify.modeling_neo_chat import NEOChatModel


class ThinkInferenceTest(unittest.TestCase):
    def test_initial_think_forward_only_computes_last_token_logits(self) -> None:
        language_model = Mock()
        model = SimpleNamespace(language_model=language_model)
        input_ids = torch.tensor([[1, 2, 3]])
        indexes = torch.zeros(3, 3, dtype=torch.long)
        attention_mask = {"full_attention": torch.zeros(1, 1, 3, 3)}

        NEOChatModel._think_prefix_forward(
            model,
            input_ids=input_ids,
            indexes=indexes,
            attention_mask=attention_mask,
        )

        language_model.assert_called_once_with(
            input_ids=input_ids,
            indexes=indexes,
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=1,
        )

    def test_appending_fixed_text_updates_cache_without_lm_head(self) -> None:
        backbone = Mock()
        embeddings = Mock(return_value=torch.zeros(1, 2, 4))
        language_model = Mock()
        language_model.model = backbone
        language_model.get_input_embeddings.return_value = embeddings
        model = SimpleNamespace(language_model=language_model)
        cache = Mock()
        cache.get_seq_length.return_value = 7
        input_ids = torch.tensor([[5, 6]])

        next_index = NEOChatModel._append_text_tokens_to_cache(model, cache, 9, input_ids)

        self.assertEqual(next_index, 11)
        backbone.assert_called_once()
        language_model.assert_not_called()
        call_kwargs = backbone.call_args.kwargs
        self.assertIs(call_kwargs["past_key_values"], cache)
        self.assertTrue(call_kwargs["use_cache"])


if __name__ == "__main__":
    unittest.main()
