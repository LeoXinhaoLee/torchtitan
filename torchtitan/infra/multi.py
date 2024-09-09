import pdb

import os
import os.path as osp
import json

import numpy as np
import torch
import pytz
from datetime import datetime
from torchtitan.infra.pytorch_utils import master_print, master_mkdir


class MultiLogger:
    def __init__(self, multi_dir, config, model_config=None):
        self.config = config
        self.model_config = model_config
        self.multi_dir = multi_dir
        self.enable = (torch.distributed.get_rank() == 0)

        master_mkdir(self.multi_dir)

        if self.enable:
            self.local_logger = open(osp.join(self.multi_dir, "config.txt"), "a")
            self._log_initial_info()

        self.all_stat_dict = {key: [] for key in ["train/loss", "learning_rate", "gradient_norm"]}

    def _log_initial_info(self):
        if self.enable:
            timezone = pytz.timezone("America/Los_Angeles")
            formatted_current_time = datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S %Z%z")

            master_print(
                f"==================================\n"
                f"Launching Time: {formatted_current_time}\n"
                f"==================================\n",
                self.local_logger,
            )

            self.local_logger.write("============= Training Config ===============\n")
            self.local_logger.write(json.dumps(self.config, indent=4, default=str) + "\n")

            if self.model_config is not None:
                self.local_logger.write("============= Model Config ===============\n")
                self.local_logger.write(json.dumps(self.model_config, indent=4) + "\n")

            self.local_logger.write("============= Training ===============\n")

    def save(self, milestone=None, ttt_stats=None):
        if self.enable:
            milestone_dir = osp.join(self.multi_dir, str(milestone)) if milestone else self.multi_dir
            master_mkdir(milestone_dir)

            if ttt_stats is not None:
                n_layer = len(ttt_stats)
                ttt_stats_dict = {
                    "ssl_tgt_last_in_mini_batch_from_mean_mse": [np.asarray(ttt_stats[i][0]) for i in range(n_layer)],
                    "ttt_loss_mse_init": [np.asarray(ttt_stats[i][1]) for i in range(n_layer)],
                    "ttt_loss_mse_step_0": [np.asarray(ttt_stats[i][2]) for i in range(n_layer)],
                    "ttt_loss_mse_step_1": [np.asarray(ttt_stats[i][3]) for i in range(n_layer)],
                }
                torch.save(ttt_stats_dict, osp.join(milestone_dir, "ttt_stats.pth"))

            torch.save(self.all_stat_dict, osp.join(milestone_dir, "all_stat_dict.pth"))

    def load(self, multi_resume_dir):
        if self.enable:
            all_stat_dict_path = osp.join(multi_resume_dir, "all_stat_dict.pth")
            if osp.exists(all_stat_dict_path):
                self.all_stat_dict = torch.load(all_stat_dict_path)

    def update_metrics(self, metrics):
        if self.enable:
            for metric in metrics:
                self.all_stat_dict["train/loss"].append(metric["train/loss"])
                self.all_stat_dict["learning_rate"].append(metric["learning_rate"])
                self.all_stat_dict["gradient_norm"].append(metric["gradient_norm"])
