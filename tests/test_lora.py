import unittest

import torch
from torch import nn

from sensenova_u1.utils.lora import load_and_merge_lora_weight


class LoRALoadingTest(unittest.TestCase):
    def test_strength_scales_the_merged_delta(self) -> None:
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data.zero_()
        state_dict = {
            "0.lora_down.weight": torch.tensor([[1.0, 2.0]]),
            "0.lora_up.weight": torch.tensor([[3.0], [4.0]]),
            "0.alpha": torch.tensor(1.0),
        }

        load_and_merge_lora_weight(model, state_dict, strength=0.5)

        torch.testing.assert_close(
            model[0].weight,
            torch.tensor([[1.5, 3.0], [2.0, 4.0]]),
        )

    def test_incompatible_lora_fails_instead_of_silently_doing_nothing(self) -> None:
        model = nn.Sequential(nn.Linear(2, 2, bias=False))

        with self.assertRaisesRegex(RuntimeError, "did not match any model weights"):
            load_and_merge_lora_weight(
                model,
                {
                    "other.lora_down.weight": torch.ones(1, 2),
                    "other.lora_up.weight": torch.ones(2, 1),
                },
            )


if __name__ == "__main__":
    unittest.main()
