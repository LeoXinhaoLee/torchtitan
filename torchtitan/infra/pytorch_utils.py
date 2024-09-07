import os
import torch
from io import BytesIO
from PIL import Image
import wandb
import matplotlib.pyplot as plt
import logging
from ml_collections import ConfigDict
from ml_collections.config_dict.config_dict import placeholder, ConfigDict


def master_mkdir(path):
    if torch.distributed.get_rank() == 0:
        os.makedirs(path, mode=0o777, exist_ok=True)


def master_print(msg, logger=None, end="\n"):
    if torch.distributed.get_rank() == 0:
        print(msg, flush=True, end=end)
        if logger is not None:
            logger.writelines(msg)
            if end == "\n":
                logger.writelines("\n")
            logger.flush()


def config_to_dict(config):
  if isinstance(config, ConfigDict) or \
     isinstance(config, dict):
    return {k: config_to_dict(v) for k, v in config.items()}
  return config

