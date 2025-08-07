from enum import Enum
from typing import Literal

from pydantic import BaseModel as Validation

Scheduler = Literal["cosine"]

Precision = Literal[
    16,
    32,
    64,
    "16",
    "16-true",
    "16-mixed",
    "bf16-true",
    "bf16-mixed",
    "32",
    "32-true",
    "64",
    "64-true",
]


class Head(str, Enum):
    Linear = "linear"
    LinearNorm = "LinearNorm"

    @staticmethod
    def needs_patches(head: str) -> bool:
        return head not in [
            Head.Linear,
            Head.LinearNorm,
        ]


class Backbone(str, Enum):
    # https://huggingface.co/docs/transformers/en/model_doc/clip
    CLIP_B_16 = "openai/clip-vit-base-patch16"
    CLIP_B_32 = "openai/clip-vit-base-patch32"
    CLIP_L_14 = "openai/clip-vit-large-patch14"
    CLIP_L_14_336 = "openai/clip-vit-large-patch14-336"


class Loss(Validation):
    # Cross-entropy loss (multi-class classification)
    ce_labels: float = 0.0  # Loss weight
    label_smoothing: float = 0.0
    # Binary cross-entropy loss (multi-label classification)
    bce_labels: float = 0.0  # Loss weight
    # Uniformity and alignment loss
    uniformity: float = 0.0  # Loss weight
    alignment_labels: float = 0.0  # Loss weight


class LoRA(Validation):
    enabled: bool = False  # Enable LoRA
    target_modules: list[str] | str = ["out_proj"]  # Target modules
    rank: int = 1  # Rank of the decomposition
    alpha: int = 32  # Scaling factor
    dropout: float = 0.1  # Dropout probability
    bias: str = "none"  # Bias configuration
    use_rslora: bool = False  # Use rsLoRA
    use_dora: bool = False  # Use DoRA


class LNTuning(Validation):
    enabled: bool = False  # Enable LayerNorm tuning
    target_modules: list[str] | str = [
        "pre_layrnorm",
        "layer_norm1",
        "layer_norm2",
        "post_layernorm",
        "layernorm",
    ]  # Target modules


class PEFT(Validation):
    enabled: bool = False  # Enable PEFT
    lora: None | LoRA = LoRA()  # LORA configuration
    ln_tuning: None | LNTuning = LNTuning()  # LayerNorm tuning configuration


class Config(Validation, validate_assignment=True):
    # Run configuration
    run_name: str = "exp-name-1"  # Name of the run
    run_dir: str = "runs/exp"  # Directory to save the run
    seed: int = 42  # Random seed for reproducibility
    throw_exception_if_run_exists: bool = False  # Throw an exception if the run directory exists

    # Model configuration
    num_classes: int = 2
    checkpoint: None | str = None  # Path to a checkpoint to load
    backbone: str = Backbone.CLIP_B_32  # Backbone model to use
    freeze_feature_extractor: bool = True  # Freeze the feature extractor
    unfreeze_layers: list[str] = []  # Layers to unfreeze
    head: str = Head.Linear  # Head model to use
    proj_feat_dim: int = 128  # Dimension of projected features
    normalize_features: bool = False  # Normalize features of penultimate layer

    # PEFT configuration
    peft: PEFT = PEFT()

    # Latent augmentations
    slerp_feature_augmentation: bool = False  # Use Slerp feature augmentation
    slerp_feature_augmentation_range: list[float] = [0.0, 1.0]  # Range of the Slerp feature augmentation

    # Data configuration
    trn_files: list[str] | dict[str, list[str]] = []  # Files containing paths to training samples
    val_files: list[str] | dict[str, list[str]] = []  # Files containing paths to validation samples
    tst_files: list[str] | dict[str, list[str]] = []  # Files containing paths to test samples
    limit_trn_files: None | int = None  # Limit the number of training files
    limit_val_files: None | int = None  # Limit the number of validation files
    limit_tst_files: None | int = None  # Limit the number of test files
    binary_labels: bool = True  # Use binary labels

    # Optimization configuration
    lr: float = 0.0003  # Learning rate (initial / base)
    min_lr: float = 1e-6  # Minimum learning rate
    lr_scheduler: None | Scheduler = "cosine"  # Learning rate scheduler
    weight_decay: float = 0.0  # AdamW weight decay
    betas: list[float] = [0.9, 0.999]  # AdamW betas
    loss: Loss = Loss()  # Loss function to use

    # Training configuration (managed by Lightning Trainer)
    max_epochs: int = 1  # Number of epochs to train
    batch_size: int = 512  # Required batch size to perform one step
    mini_batch_size: int = 512  # Mini batch size per device
    num_workers: int = 12  # Number of workers for the DataLoader
    devices: list[int] | str | int = "auto"  # Devices to use for training
    precision: Precision = "bf16-mixed"  # Precision for the model
    fast_dev_run: int | bool = False  # Run a fast development run
    overfit_batches: int | float = 0.0  # Overfit on a subset of the data
    limit_train_batches: None | int | float = None  # Limit the number of training batches
    limit_test_batches: None | int | float = None  # Limit the number of test batches
    limit_val_batches: None | int | float = None  # Limit the number of validation batches
    deterministic: None | bool = None  # Set random seed for reproducibility
    detect_anomaly: bool = False  # Detect anomalies in the model
    checkpoint_for_testing: str = "best_mAP"  # Checkpoint to use for testing

    # Logging
    wandb: bool = False  # Log metrics to Weights & Biases
    wandb_tags: list[str] = []  # Tags to use for Weights & Biases

    # Post-processing
    make_binary_before_video_aggregation: bool = True  # Make binary labels before video aggregation


def load_config(path: str) -> Config:
    import yaml

    # read yaml config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # overwrite config
    config = Config(**config)
    return config
