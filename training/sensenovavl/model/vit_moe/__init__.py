from sensenovavl.model.vit_moe.gshard_layer import InternVitGShardMoELayer


def new_moe_layer(moe_type: str, **kwargs):
    if moe_type == "GShard":
        return InternVitGShardMoELayer(**kwargs)


__all__ = ["new_moe_layer"]
