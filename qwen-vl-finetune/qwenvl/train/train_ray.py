"""
Ray Train wrapper for Qwen2.5-VL training.

This module runs the same training as train_qwen.py but using Ray Train for
multi-node distributed training on Anyscale clusters. Ray Train handles
the distributed setup (MASTER_ADDR, node ranks, etc.) automatically.

The key simplification: this wrapper just calls the existing train() function
from train_qwen.py, passing through all arguments. This ensures identical
behavior and memory usage between Ray and non-Ray versions.

Usage:
    python -m qwenvl.train.train_ray \
        --model_name_or_path Qwen/Qwen2.5-VL-32B-Instruct \
        --dataset_use mscoco2017_val_captions \
        --max_steps 100 \
        --num_workers 8 \
        ...
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

import ray
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer


def train_loop(config):
    """
    Training loop that runs on each Ray worker.

    Simply calls the train() function from train_qwen.py with the
    appropriate sys.argv set up for HfArgumentParser.

    Ray Train sets up the distributed environment (MASTER_ADDR, MASTER_PORT,
    RANK, WORLD_SIZE, LOCAL_RANK) automatically, and HuggingFace's Trainer
    + DeepSpeed will use these to initialize distributed training.
    """
    import torch.distributed as dist

    # Add project root to path for imports
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Ray Train may have already initialized the process group
    # If so, we need to destroy it so DeepSpeed/HuggingFace can reinitialize
    # if dist.is_initialized():
    #     dist.destroy_process_group()

    # Build sys.argv from config for HfArgumentParser
    argv = ["train_qwen.py"]  # Script name (not used but required)

    # Add all training arguments
    for key, value in config.items():
        if key.startswith("_"):  # Skip internal keys
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                argv.append(f"--{key}")
        else:
            argv.append(f"--{key}")
            argv.append(str(value))

    # Replace sys.argv so HfArgumentParser works correctly
    original_argv = sys.argv
    sys.argv = argv

    try:
        # Import and call train() from train_qwen.py
        from qwenvl.train.train_qwen import train
        train(attn_implementation="flash_attention_2")
    finally:
        sys.argv = original_argv


def main():
    """Main entry point for Ray Train Qwen2.5-VL training."""
    parser = argparse.ArgumentParser(description="Ray Train wrapper for Qwen2.5-VL")

    # Ray Train specific arguments
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of Ray workers (typically one per GPU)")
    parser.add_argument("--storage_path", type=str, default="/mnt/cluster_storage",
                        help="Storage path for checkpoints")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name for the training experiment")

    # All other arguments will be passed through to train_qwen.py
    # We use parse_known_args to capture Ray-specific args and pass the rest through
    args, remaining_args = parser.parse_known_args()

    # Parse the remaining arguments to build the config dict
    # These are the arguments that will be passed to train_qwen.py
    train_parser = argparse.ArgumentParser()

    # Model arguments
    train_parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct")
    train_parser.add_argument("--tune_mm_vision", type=str, default="True")
    train_parser.add_argument("--tune_mm_mlp", type=str, default="True")
    train_parser.add_argument("--tune_mm_llm", type=str, default="True")

    # Data arguments
    train_parser.add_argument("--dataset_use", type=str, default="mscoco2017_val_captions")
    train_parser.add_argument("--data_flatten", type=str, default="True")
    train_parser.add_argument("--data_packing", type=str, default=None)
    train_parser.add_argument("--max_pixels", type=int, default=200704)
    train_parser.add_argument("--min_pixels", type=int, default=200704)
    train_parser.add_argument("--force_fixed_size", type=str, default="True")
    train_parser.add_argument("--model_max_length", type=int, default=8192)

    # Training arguments (these map to TrainingArguments in train_qwen.py)
    train_parser.add_argument("--output_dir", type=str, default="/tmp/qwen-vl-output")
    train_parser.add_argument("--max_steps", type=int, default=100)
    train_parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    train_parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    train_parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    train_parser.add_argument("--learning_rate", type=float, default=2e-7)
    train_parser.add_argument("--weight_decay", type=float, default=0.0)
    train_parser.add_argument("--warmup_steps", type=int, default=3)
    train_parser.add_argument("--max_grad_norm", type=float, default=1.0)
    train_parser.add_argument("--logging_steps", type=int, default=1)
    train_parser.add_argument("--gradient_checkpointing", type=str, default="True")
    train_parser.add_argument("--bf16", action="store_true", default=True)
    train_parser.add_argument("--eval_strategy", type=str, default="no")
    train_parser.add_argument("--save_strategy", type=str, default="no")
    train_parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    train_parser.add_argument("--dataloader_num_workers", type=int, default=4)
    train_parser.add_argument("--run_name", type=str, default="qwen2vl-ray-train")
    train_parser.add_argument("--report_to", type=str, default="none")
    train_parser.add_argument("--cache_dir", type=str, default=None)

    # DeepSpeed config
    train_parser.add_argument("--deepspeed", type=str, default=None)

    train_args, _ = train_parser.parse_known_args(remaining_args)

    print(f"Ray Train Arguments: num_workers={args.num_workers}, storage_path={args.storage_path}")
    print(f"Training Arguments: {train_args}")

    # Build config dict for train_loop
    train_loop_config = {}
    for key, value in vars(train_args).items():
        if value is not None:
            train_loop_config[key] = value

    # If no deepspeed config provided, use the default zero3.json
    if train_args.deepspeed is None:
        default_ds_config = Path(__file__).resolve().parent.parent.parent / "scripts" / "zero3.json"
        if default_ds_config.exists():
            train_loop_config["deepspeed"] = str(default_ds_config)
            print(f"Using default DeepSpeed config: {default_ds_config}")

    # Initialize Ray if not already connected
    if not ray.is_initialized():
        ray.init()

    # Scaling config
    scaling_config = ScalingConfig(
        num_workers=args.num_workers,
        use_gpu=True,
    )

    # Run config
    experiment_name = args.experiment_name or train_args.run_name
    run_config = RunConfig(
        storage_path=args.storage_path,
        name=experiment_name,
    )

    # Create trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    print(f"Starting Ray Train with {args.num_workers} workers...")
    print(f"Experiment name: {experiment_name}")

    result = trainer.fit()

    print(f"Training finished. Result: {result}")

    return result


if __name__ == "__main__":
    main()
