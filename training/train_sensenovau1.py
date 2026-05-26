# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Main training entry point for SenseNova-U1.
import logging
import os
import time

import torch

from sensenovalm.accelerator import get_accelerator
from sensenovalm.core.context import global_context as gpc
from sensenovalm.core.trainer_builder import TrainerBuilder
from sensenovalm.initialize import initialize_distributed_env
from sensenovalm.utils.common import get_current_device, parse_args
from sensenovavl.data import build_train_loader_with_data_type
from sensenovavl.train.pipeline import get_model
from sensenovavl.utils.utils import init_pil

# global llm logger
logger = logging.getLogger(__file__)
sensenovalm_accelerator = get_accelerator()


def patch_inductor_triton_max_block():
    max_block_x = os.environ.get("TORCHINDUCTOR_TRITON_MAX_BLOCK_X")
    if not max_block_x:
        return

    try:
        max_block_x = int(max_block_x)
    except ValueError:
        logger.warning("Invalid TORCHINDUCTOR_TRITON_MAX_BLOCK_X=%s, skip patch.", max_block_x)
        return

    try:
        from torch._inductor.runtime.hints import TRITON_MAX_BLOCK
    except Exception:
        logger.warning("Failed to import torch._inductor.runtime.hints.TRITON_MAX_BLOCK.", exc_info=True)
        return

    old_max_block_x = TRITON_MAX_BLOCK.get("X")
    if old_max_block_x is None or max_block_x > old_max_block_x:
        TRITON_MAX_BLOCK["X"] = max_block_x
        logger.info("Patched TRITON_MAX_BLOCK['X'] from %s to %s.", old_max_block_x, max_block_x)


def main(args):
    if gpc.config.get("MP_SPAWN", False):
        torch.multiprocessing.set_start_method("spawn")

    very_beginning_time = time.time()

    # initialize the train and validation data loader
    train_dl, dataset_types = build_train_loader_with_data_type()
    val_dls = None

    # get sensenovavl model
    model = get_model(gpc.config.model, gpc.config.data).to(get_current_device())

    # build trainer
    merged_args = {
        **vars(args),
        "dataset_types": dataset_types,
        "very_begining_time": very_beginning_time,
    }

    trainer = TrainerBuilder(
        model,
        train_dl,
        val_dls,
        **merged_args,
    )

    # training
    trainer.fit()


if __name__ == "__main__":
    patch_inductor_triton_max_block()

    args = parse_args()

    # initialize distributed environment
    initialize_distributed_env(config=args.config, launcher=args.launcher, master_port=args.port, seed=args.seed)
    assert hasattr(gpc, "config") and gpc.config is not None

    init_pil()

    # Run the main function with parsed arguments
    main(args)
