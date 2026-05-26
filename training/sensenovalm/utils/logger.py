#!/usr/bin/env python
# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
# -*- encoding: utf-8 -*-

import logging
import os

LOGGER_NAME = "sensenovalm"
LOGGER_FORMAT = "%(asctime)s\t%(levelname)s %(filename)s:%(lineno)s in %(funcName)s -- %(message)s"
LOGGER_LEVEL = "info"
LOGGER_LEVEL_CHOICES = ["debug", "info", "warning", "error", "critical"]
LOGGER_LEVEL_HELP = (
    "The logging level threshold, choices=['debug', 'info', 'warning', 'error', 'critical'], default='info'"
)

std_logger = None


def get_logger(
    logger_name: str = LOGGER_NAME,
    logging_level: str = LOGGER_LEVEL,
    launch_time: str = None,
    job_name: str = None,
    file_name: str = None,
) -> logging.Logger:
    """Configure the logger that is used for uniscale framework.

    Args:
        logger_name (str): used to create or get the correspoding logger in
            getLogger call. It will be "sensenovalm" by default.
        logging_level (str, optional): Logging level in string or logging enum.

    Returns:
        logger (logging.Logger): the created or modified logger.

    """
    global std_logger
    if std_logger is not None:
        return std_logger

    logger = logging.getLogger(logger_name)

    if logging_level not in LOGGER_LEVEL_CHOICES:
        logging_level = LOGGER_LEVEL
        print(LOGGER_LEVEL_HELP)

    logging_level = logging.getLevelName(logging_level.upper())

    # add stream handler
    handler = logging.StreamHandler()
    handler.setLevel(logging_level)
    logger.setLevel(logging_level)
    handler.setFormatter(logging.Formatter(LOGGER_FORMAT))
    logger.addHandler(handler)

    # add file handler
    if file_name is not None:
        log_folder = os.path.join("RUN", job_name, launch_time, "logs")
        log_filepath = os.path.join(log_folder, file_name)
        try:
            os.makedirs(log_folder, exist_ok=True)
        except FileExistsError:
            pass
        filehandler = logging.FileHandler(log_filepath)
        filehandler.setLevel(logging_level)
        filehandler.setFormatter(logging.Formatter(LOGGER_FORMAT))
        logger.addHandler(filehandler)

        std_logger = logger

    return logger
