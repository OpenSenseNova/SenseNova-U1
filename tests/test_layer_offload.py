import unittest

import torch
from torch import nn

from sensenova_u1.utils.layer_offload import (
    _ALL_TENSOR_GROUPS,
    _GROUP_GENERATION,
    _GROUP_SHARED,
    _GROUP_UNDERSTANDING,
    _partition_layer_tensor_names,
    _required_tensor_groups,
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


if __name__ == "__main__":
    unittest.main()
