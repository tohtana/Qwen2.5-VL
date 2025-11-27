"""
Ray Train wrapper for Qwen2.5-VL training.

This module runs the same training as train_qwen.py but using Ray Train for
multi-node distributed training on Anyscale clusters. Ray Train handles
the distributed setup (MASTER_ADDR, node ranks, etc.) automatically.

Usage:
    python -m qwenvl.train.train_ray \
        --model_name_or_path Qwen/Qwen2.5-VL-32B-Instruct \
        --dataset_use mscoco2017_val_captions \
        --max_steps 100 \
        --num_workers 8 \
        ...
"""

import argparse
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

import deepspeed
import ray
import ray.train
import ray.train.torch
import torch
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from torch.utils.data import DataLoader

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


def log_rank0(message: str) -> None:
    """Log message only on rank 0."""
    if ray.train.get_context().get_world_rank() == 0:
        logger.info(message)


def setup_model(config: Dict[str, Any]) -> torch.nn.Module:
    """
    Load and configure the Qwen2.5-VL model.

    Args:
        config: Training configuration dictionary.

    Returns:
        Configured model ready for training.
    """
    from transformers import (
        Qwen2VLForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
    )

    try:
        from transformers import (
            Qwen3VLForConditionalGeneration,
            Qwen3VLMoeForConditionalGeneration,
        )
        QWEN3_VL_AVAILABLE = True
    except ImportError:
        QWEN3_VL_AVAILABLE = False
        Qwen3VLForConditionalGeneration = None
        Qwen3VLMoeForConditionalGeneration = None

    model_name = config["model_name_or_path"]
    attn_implementation = config.get("attn_implementation", "flash_attention_2")
    bf16 = config.get("bf16", True)
    cache_dir = config.get("cache_dir", None)

    model_path_lower = model_name.lower()

    if "qwen3" in model_path_lower and "moe" in model_path_lower:
        if not QWEN3_VL_AVAILABLE:
            raise ImportError("Qwen3VL models not available in this transformers version.")
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16 if bf16 else None,
        )
        model_type = "qwen3vl"
    elif "qwen3" in model_path_lower:
        if not QWEN3_VL_AVAILABLE:
            raise ImportError("Qwen3VL models not available in this transformers version.")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16 if bf16 else None,
        )
        model_type = "qwen3vl"
    elif "qwen2.5" in model_path_lower:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16 if bf16 else None,
        )
        model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16 if bf16 else None,
        )
        model_type = "qwen2vl"

    log_rank0(f"Loaded model: {model_name}, class: {model.__class__.__name__}")
    log_rank0(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Configure trainable parameters
    tune_mm_vision = config.get("tune_mm_vision", True)
    tune_mm_mlp = config.get("tune_mm_mlp", True)
    tune_mm_llm = config.get("tune_mm_llm", True)

    for n, p in model.visual.named_parameters():
        p.requires_grad = tune_mm_vision

    for n, p in model.visual.merger.named_parameters():
        p.requires_grad = tune_mm_mlp

    for n, p in model.model.named_parameters():
        p.requires_grad = tune_mm_llm

    if hasattr(model, "lm_head"):
        model.lm_head.requires_grad = tune_mm_llm

    model.config.use_cache = False

    # Enable gradient checkpointing if requested
    if config.get("gradient_checkpointing", True):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    return model, model_type


def setup_dataloader(config: Dict[str, Any], model_type: str) -> DataLoader:
    """
    Set up the data loader for training.

    Args:
        config: Training configuration dictionary.
        model_type: Type of model (qwen2vl, qwen2.5vl, qwen3vl).

    Returns:
        DataLoader for training data.
    """
    from transformers import AutoProcessor
    from qwenvl.data.data_processor import make_supervised_data_module

    # Create a simple data args object
    class DataArgs:
        pass

    data_args = DataArgs()
    data_args.dataset_use = config.get("dataset_use", "mscoco2017_val_captions")
    data_args.data_flatten = config.get("data_flatten", True)
    data_args.data_packing = config.get("data_packing", False)
    data_args.base_interval = config.get("base_interval", 2)
    data_args.max_pixels = config.get("max_pixels", 200704)
    data_args.min_pixels = config.get("min_pixels", 200704)
    data_args.video_max_frames = config.get("video_max_frames", 8)
    data_args.video_min_frames = config.get("video_min_frames", 4)
    data_args.video_max_pixels = config.get("video_max_pixels", 1024 * 28 * 28)
    data_args.video_min_pixels = config.get("video_min_pixels", 256 * 28 * 28)
    data_args.video_fps = config.get("video_fps", 2)
    data_args.force_fixed_size = config.get("force_fixed_size", True)
    data_args.model_type = model_type

    # Load processor
    model_name = config["model_name_or_path"]
    processor = AutoProcessor.from_pretrained(model_name)

    # Create data module
    data_module = make_supervised_data_module(processor, data_args=data_args)

    # Create DataLoader
    batch_size = config.get("per_device_train_batch_size", 1)
    num_workers = config.get("dataloader_num_workers", 4)

    train_loader = DataLoader(
        data_module["train_dataset"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_module["data_collator"],
        num_workers=num_workers,
        pin_memory=True,
    )

    return ray.train.torch.prepare_data_loader(train_loader)


def report_metrics_and_save_checkpoint(
    ds_engine: deepspeed.runtime.engine.DeepSpeedEngine,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """
    Report training metrics and optionally save checkpoint.

    Args:
        ds_engine: DeepSpeed engine.
        metrics: Dictionary of metrics to report.
        config: Training configuration.
    """
    ctx = ray.train.get_context()

    if config.get("save_strategy", "no") == "no":
        # Just report metrics, no checkpoint
        ray.train.report(metrics)
        return

    step = metrics.get("step", 0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_dir = os.path.join(tmp_dir, "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)

        ds_engine.save_checkpoint(checkpoint_dir)

        # Save step info
        step_file = os.path.join(checkpoint_dir, "step.txt")
        with open(step_file, "w", encoding="utf-8") as f:
            f.write(str(step))

        checkpoint = Checkpoint.from_directory(tmp_dir)
        ray.train.report(metrics, checkpoint=checkpoint)

        if ctx.get_world_rank() == 0:
            log_rank0(f"Checkpoint saved at step {step}. Metrics: {metrics}")


def load_checkpoint(
    ds_engine: deepspeed.runtime.engine.DeepSpeedEngine,
    ckpt: ray.train.Checkpoint,
) -> int:
    """
    Load checkpoint and return the next step to resume from.

    Args:
        ds_engine: DeepSpeed engine.
        ckpt: Ray Train checkpoint.

    Returns:
        Next step number to resume training from.
    """
    next_step = 0
    try:
        with ckpt.as_directory() as checkpoint_dir:
            log_rank0(f"Loading checkpoint from {checkpoint_dir}")
            ckpt_dir = os.path.join(checkpoint_dir, "checkpoint")
            if not os.path.isdir(ckpt_dir):
                ckpt_dir = checkpoint_dir

            ds_engine.load_checkpoint(ckpt_dir)

            step_file = os.path.join(ckpt_dir, "step.txt")
            if os.path.isfile(step_file):
                with open(step_file, "r", encoding="utf-8") as f:
                    last_step = int(f.read().strip())
                next_step = last_step + 1

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()

        log_rank0(f"Successfully loaded checkpoint, resuming from step {next_step}")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise RuntimeError(f"Checkpoint loading failed: {e}") from e

    return next_step


def train_loop(config: Dict[str, Any]) -> None:
    """
    Main training loop that runs on each Ray worker.

    This function implements the training loop using DeepSpeed for distributed
    training. Ray Train handles the distributed setup automatically.

    Args:
        config: Dictionary containing all training configuration.
    """
    # Add project root to path for imports (must be done in each worker)
    # This is necessary because Ray workers don't inherit sys.path from the driver
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    # Also add to handle Anyscale job packaging (working_dir is at qwen-vl-finetune level)
    finetune_root = project_root / "qwen-vl-finetune"
    if finetune_root.exists() and str(finetune_root) not in sys.path:
        sys.path.insert(0, str(finetune_root))

    # Replace attention class for data flattening support
    from qwenvl.train.trainer import replace_qwen2_vl_attention_class

    if config.get("data_flatten", True) or config.get("data_packing", False):
        replace_qwen2_vl_attention_class()

    # Setup model
    model, model_type = setup_model(config)

    log_rank0(f"Model type: {model_type}")

    # Setup optimizer
    learning_rate = config.get("learning_rate", 2e-7)
    weight_decay = config.get("weight_decay", 0.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # DeepSpeed configuration
    ds_config = config.get("ds_config", {
        "train_micro_batch_size_per_gpu": config.get("per_device_train_batch_size", 1),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 1),
        "bf16": {"enabled": config.get("bf16", True)},
        "grad_accum_dtype": "bf16" if config.get("bf16", True) else "fp32",
        "zero_optimization": {
            "stage": config.get("zero_stage", 3),
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "gradient_clipping": config.get("max_grad_norm", 1.0),
    })

    # Initialize DeepSpeed
    ds_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
    )

    # Load checkpoint if exists
    ckpt = ray.train.get_checkpoint()
    start_step = 0
    if ckpt:
        start_step = load_checkpoint(ds_engine, ckpt)
        log_rank0(f"Resuming training from step {start_step}")

    # Setup data loader
    train_loader = setup_dataloader(config, model_type)
    device = ray.train.torch.get_device()

    # Training configuration
    max_steps = config.get("max_steps", 100)
    warmup_steps = config.get("warmup_steps", 3)
    logging_steps = config.get("logging_steps", 1)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)

    # Set model to training mode
    ds_engine.train()

    # Training loop
    log_rank0(f"Starting training: max_steps={max_steps}, warmup_steps={warmup_steps}")

    iteration_times = []
    global_step = start_step
    running_loss = 0.0
    num_batches = 0

    data_iter = iter(train_loader)

    while global_step < max_steps:
        # Get next batch (cycle through data if needed)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        measure_time = global_step >= warmup_steps
        step_start = time.perf_counter() if measure_time else None

        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        position_ids = batch.get("position_ids")
        if position_ids is not None:
            position_ids = position_ids.to(device)

        # Handle pixel values
        pixel_values = batch.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)

        image_grid_thw = batch.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device)

        pixel_values_videos = batch.get("pixel_values_videos")
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.to(device)

        video_grid_thw = batch.get("video_grid_thw")
        if video_grid_thw is not None:
            video_grid_thw = video_grid_thw.to(device)

        # Forward pass
        outputs = ds_engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            labels=labels,
            use_cache=False,
        )

        loss = outputs.loss

        # Backward pass
        ds_engine.backward(loss)
        ds_engine.step()

        running_loss += loss.item()
        num_batches += 1

        # Record timing
        if measure_time and step_start is not None:
            torch.cuda.synchronize()
            step_elapsed = time.perf_counter() - step_start
            iteration_times.append(step_elapsed)

        # Logging
        if (global_step + 1) % logging_steps == 0 or global_step == 0:
            status = "warmup" if global_step < warmup_steps else "training"
            log_rank0(
                f"Step {global_step + 1}/{max_steps} ({status}) - "
                f"Loss: {loss.item():.4f}"
            )

        global_step += 1

    # Report final metrics
    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    if iteration_times:
        avg_time = sum(iteration_times) / len(iteration_times)
        steps_per_sec = 1.0 / avg_time if avg_time > 0 else 0
        log_rank0("=" * 60)
        log_rank0("TRAINING COMPLETED - TIMING STATISTICS")
        log_rank0("=" * 60)
        log_rank0(f"Total steps: {global_step}")
        log_rank0(f"Warmup steps: {warmup_steps}")
        log_rank0(f"Measured steps: {len(iteration_times)}")
        log_rank0(f"Average iteration time: {avg_time:.4f}s")
        log_rank0(f"Steps per second: {steps_per_sec:.4f}")
        log_rank0("=" * 60)

        # Print in format expected by run_sweep.sh
        print(f"Training completed! {max_steps} total steps, "
              f"{len(iteration_times)} measured steps (avg {avg_time:.3f}s/step)")

    # report_metrics_and_save_checkpoint(
    #     ds_engine,
    #     {
    #         "loss": avg_loss,
    #         "step": global_step,
    #         "avg_iter_time": avg_time if iteration_times else 0,
    #         "steps_per_sec": steps_per_sec if iteration_times else 0,
    #     },
    #     config,
    # )


def main():
    """Main entry point for Ray Train Qwen2.5-VL training."""
    parser = argparse.ArgumentParser(description="Ray Train wrapper for Qwen2.5-VL")

    # Ray Train arguments
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of Ray workers (typically one per GPU)")
    parser.add_argument("--storage_path", type=str, default="/mnt/cluster_storage",
                        help="Storage path for checkpoints")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name for the training experiment")
    parser.add_argument("--resume_experiment", type=str, default=None,
                        help="Experiment name to resume from")

    # Model arguments
    parser.add_argument("--model_name_or_path", type=str,
                        default="Qwen/Qwen2.5-VL-32B-Instruct",
                        help="Path to pretrained model")
    parser.add_argument("--tune_mm_vision", type=lambda x: x.lower() == "true",
                        default=True, help="Finetune vision encoder")
    parser.add_argument("--tune_mm_mlp", type=lambda x: x.lower() == "true",
                        default=True, help="Finetune MLP projector")
    parser.add_argument("--tune_mm_llm", type=lambda x: x.lower() == "true",
                        default=True, help="Finetune language model")
    parser.add_argument("--attn_implementation", type=str,
                        default="flash_attention_2",
                        help="Attention implementation")

    # Data arguments
    parser.add_argument("--dataset_use", type=str, default="mscoco2017_val_captions",
                        help="Dataset name to use")
    parser.add_argument("--data_flatten", type=lambda x: x.lower() == "true",
                        default=True, help="Use data flattening")
    parser.add_argument("--max_pixels", type=int, default=200704,
                        help="Maximum pixels for images")
    parser.add_argument("--min_pixels", type=int, default=200704,
                        help="Minimum pixels for images")
    parser.add_argument("--force_fixed_size", type=lambda x: x.lower() == "true",
                        default=True, help="Force fixed image size")
    parser.add_argument("--model_max_length", type=int, default=8192,
                        help="Maximum sequence length")

    # Training arguments
    parser.add_argument("--max_steps", type=int, default=100,
                        help="Maximum training steps")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-7,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=3,
                        help="Number of warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="Log every N steps")
    parser.add_argument("--gradient_checkpointing", type=lambda x: x.lower() == "true",
                        default=True, help="Use gradient checkpointing")
    parser.add_argument("--bf16", type=lambda x: x.lower() == "true",
                        default=True, help="Use bfloat16 training")

    # DeepSpeed arguments
    parser.add_argument("--zero_stage", type=int, default=3,
                        help="DeepSpeed ZeRO stage (1, 2, or 3)")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="Path to DeepSpeed config JSON (optional)")

    # Misc
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--output_dir", type=str, default="/tmp/qwen-vl-output",
                        help="Output directory")
    parser.add_argument("--save_strategy", type=str, default="no",
                        help="Checkpoint save strategy")
    parser.add_argument("--run_name", type=str, default="qwen2vl-ray-train",
                        help="Name for this training run")
    parser.add_argument("--report_to", type=str, default="none",
                        help="Where to report metrics")

    args = parser.parse_args()

    print(f"Arguments: {args}")

    # Initialize Ray if not already connected to a cluster
    if not ray.is_initialized():
        ray.init()

    # Build training config
    train_loop_config = {
        # Model
        "model_name_or_path": args.model_name_or_path,
        "tune_mm_vision": args.tune_mm_vision,
        "tune_mm_mlp": args.tune_mm_mlp,
        "tune_mm_llm": args.tune_mm_llm,
        "attn_implementation": args.attn_implementation,
        "gradient_checkpointing": args.gradient_checkpointing,
        # Data
        "dataset_use": args.dataset_use,
        "data_flatten": args.data_flatten,
        "max_pixels": args.max_pixels,
        "min_pixels": args.min_pixels,
        "force_fixed_size": args.force_fixed_size,
        "model_max_length": args.model_max_length,
        # Training
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": args.max_grad_norm,
        "logging_steps": args.logging_steps,
        "bf16": args.bf16,
        # DeepSpeed
        "zero_stage": args.zero_stage,
        # Misc
        "dataloader_num_workers": args.dataloader_num_workers,
        "output_dir": args.output_dir,
        "save_strategy": args.save_strategy,
    }

    # Load custom DeepSpeed config if provided
    if args.deepspeed:
        import json
        with open(args.deepspeed, "r") as f:
            train_loop_config["ds_config"] = json.load(f)

    # Scaling config
    scaling_config = ScalingConfig(
        num_workers=args.num_workers,
        use_gpu=True,
    )

    # Run config
    experiment_name = args.experiment_name or args.resume_experiment or args.run_name
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
