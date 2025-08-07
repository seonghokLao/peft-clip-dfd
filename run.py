import os
from glob import glob

import fire
import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rich
import torch
import torch.nn as nn
import yaml
from lightning import Trainer
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_loggers
from rich import traceback

from src import dataset as datasets
from src import model as models
from src.config import Backbone, Config, Head, load_config
from src.utils import files
from src.utils.checks import checks
from src.utils.model_checkpoint import ModelCheckpointParallel

traceback.install()


def main(config: Config, train: bool):
    checks(config)

    torch.set_float32_matmul_precision("high")  # Set the precision for matmul operations

    model = models.DeepfakeDetectionModel(config, verbose=True)

    if config.checkpoint:
        model.load_state_dict(torch.load(config.checkpoint, map_location="cpu", weights_only=True)["state_dict"])

    data_module = datasets.DeepfakeDataModule(config, model.get_preprocessing())

    loggers: list = [pl_loggers.CSVLogger(config.run_dir, name=config.run_name, version="")]

    if config.wandb:
        wandb_logger = pl_loggers.WandbLogger(
            project="deepfake",
            name=config.run_name,
            save_dir=f"{config.run_dir}/{config.run_name}",
            tags=config.wandb_tags,
        )
        loggers.append(wandb_logger)

    callbacks = [
        pl_callbacks.RichProgressBar(),
        ModelCheckpointParallel(filename="best_mAP", monitor="val/mAP_video", mode="max"),
    ]

    trainer = Trainer(
        devices=config.devices,
        max_epochs=config.max_epochs,
        precision=config.precision,
        accumulate_grad_batches=config.batch_size // config.mini_batch_size,
        fast_dev_run=config.fast_dev_run,
        log_every_n_steps=100,
        overfit_batches=config.overfit_batches,
        limit_train_batches=config.limit_train_batches,
        limit_val_batches=config.limit_val_batches,
        limit_test_batches=config.limit_test_batches,
        deterministic=config.deterministic,
        detect_anomaly=config.detect_anomaly,
        logger=loggers,
        callbacks=callbacks,
        default_root_dir=config.run_dir,
    )

    if train:
        trainer.fit(model, data_module)

        ckpt_path = f"{config.run_dir}/{config.run_name}/checkpoints/{config.checkpoint_for_testing}.ckpt"
        trainer.test(model, data_module, ckpt_path=ckpt_path)

    else:
        assert config.checkpoint is not None, "Checkpoint is required for testing"
        trainer.test(model, data_module)

    if config.wandb:
        wandb_logger.finalize("success")
        wandb_logger.experiment.finish()


def get_train_config() -> Config:
    config = Config()

    config.run_name = "example-run"
    config.run_dir = "runs/train"
    config.wandb = False

    config.num_workers = 12
    config.devices = [2]

    config.backbone = Backbone.CLIP_L_14
    config.freeze_feature_extractor = True
    config.peft.enabled = True
    config.peft.ln_tuning.enabled = True
    config.head = Head.LinearNorm
    config.num_classes = 2
    config.loss.ce_labels = 1.0
    config.slerp_feature_augmentation = True

    config.batch_size = config.mini_batch_size = 128
    config.lr_scheduler = "cosine"
    config.lr = 8e-5
    config.min_lr = 5e-5
    config.weight_decay = 0
    config.max_epochs = 10

    limit_val_files = 16384
    config.limit_val_files = limit_val_files
    config.limit_val_batches = limit_val_files // config.mini_batch_size

    config.binary_labels = True
    config.trn_files = [
        "config/datasets/FF/test/DF.txt",
        "config/datasets/FF/test/F2F.txt",
        "config/datasets/FF/test/FS.txt",
        "config/datasets/FF/test/NT.txt",
        "config/datasets/FF/test/real.txt",
    ]
    config.val_files = [
        "config/datasets/CDFv2/test/Celeb-synthesis.txt",
        "config/datasets/CDFv2/test/Celeb-real.txt",
        "config/datasets/CDFv2/test/YouTube-real.txt",
    ]

    config.tst_files = {
        "CDF": [
            "config/datasets/CDFv2/test/Celeb-synthesis.txt",
            "config/datasets/CDFv2/test/Celeb-real.txt",
            "config/datasets/CDFv2/test/YouTube-real.txt",
        ]
    }

    return config


def get_test_config() -> Config:
    config_path = "runs/train/example-run/hparams.yaml"
    new_run_name = "example-run"

    config = load_config(config_path)

    config.run_name = new_run_name
    config.run_dir = "runs/test"
    config.checkpoint = config_path.replace("hparams.yaml", "checkpoints/best_mAP.ckpt")
    config.wandb = False
    config.wandb_tags.extend(["test"])

    config.num_workers = 12
    config.batch_size = config.mini_batch_size = 512
    config.devices = [0]

    config.tst_files = {
        "CDF": [
            "config/datasets/CDFv2/test/Celeb-synthesis.txt",
            "config/datasets/CDFv2/test/Celeb-real.txt",
            "config/datasets/CDFv2/test/YouTube-real.txt",
        ]
    }

    return config


def get_debug_config(config: Config) -> Config:
    #! Debug

    config.run_name = "tmp"

    config.devices = [2]

    config.num_workers = 8
    # config.batch_size = config.mini_batch_size = 512
    config.max_epochs = 1
    config.limit_train_batches = 12
    config.limit_val_batches = 12
    config.limit_test_batches = 12
    config.deterministic = True
    config.detect_anomaly = True

    return config


def entry(train: bool = False, test: bool = False, debug: bool = False, **kwargs):
    if train:
        config = get_train_config()

    elif test:
        config = get_test_config()

    else:
        raise ValueError("Either --train or --test must be provided")

    # Overwrite config with debug values
    if debug:
        config = config.model_copy(update=dict(get_debug_config(config)))

    # Parse command line arguments
    config = config.model_copy(update=kwargs)

    # Revalidate the config - checks if user provided valid values
    config = Config(**dict(config))

    main(config, train)


if __name__ == "__main__":
    fire.Fire(entry)
