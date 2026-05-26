import torch
import torch.distributed as dist

from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.utils.common import get_current_device


def padding_images(images=None):
    cur_num_images = 0
    num_padding_images = 0
    if images is not None:
        cur_num_images = images.shape[0]

    max_num_images = torch.tensor([cur_num_images], dtype=torch.long, device=get_current_device())
    dist.all_reduce(max_num_images, op=dist.ReduceOp.MAX, group=gpc.get_group(ParallelMode.DATA))
    if max_num_images > 0:
        num_padding_images = max_num_images.item() - cur_num_images
        if num_padding_images > 0:
            image_size = gpc.config.data.force_image_size
            padding_images = torch.zeros((num_padding_images, 3, image_size, image_size), dtype=torch.float32)
            if images is None:
                images = padding_images
            else:
                images = torch.cat((images, padding_images), dim=0)
    image_flags = [1] * cur_num_images + [0] * num_padding_images
    image_flags = torch.LongTensor(image_flags)

    return images, image_flags
