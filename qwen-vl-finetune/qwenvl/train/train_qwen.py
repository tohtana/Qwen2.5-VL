# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.train.trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

try:
    from transformers import (
        Qwen3VLForConditionalGeneration,
        Qwen3VLMoeForConditionalGeneration
    )
    QWEN3_VL_AVAILABLE = True
except ImportError:
    QWEN3_VL_AVAILABLE = False
    Qwen3VLForConditionalGeneration = None
    Qwen3VLMoeForConditionalGeneration = None
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer, TrainerCallback

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


class QwenVLTrainer(Trainer):
    """Custom Trainer with additional logging for input shapes and accurate iteration timing"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iteration_times = []
        self.step_start_time = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training step to log input shapes and record accurate iteration time
        """
        # Record start time for this iteration (after data loading)
        if self.args.local_rank in [-1, 0]:
            torch.cuda.synchronize()  # Ensure GPU operations are complete
            self.step_start_time = time.time()

        # Log shapes on rank 0 only
        if self.args.local_rank in [-1, 0]:
            input_ids = inputs.get('input_ids')
            pixel_values = inputs.get('pixel_values')
            image_grid_thw = inputs.get('image_grid_thw')
            attention_mask = inputs.get('attention_mask')

            log_parts = []
            log_parts.append(f"input_ids={tuple(input_ids.shape) if input_ids is not None else None}")
            if pixel_values is not None:
                log_parts.append(f"pixel_values={tuple(pixel_values.shape)}")
            if image_grid_thw is not None:
                log_parts.append(f"image_grid_thw={tuple(image_grid_thw.shape)}")
            if attention_mask is not None:
                log_parts.append(f"attention_mask={tuple(attention_mask.shape)}")
            # print(", ".join(log_parts))

        # Call parent training_step
        loss = super().training_step(model, inputs, num_items_in_batch)

        # Record end time for this iteration (after backward pass and optimizer step)
        if self.args.local_rank in [-1, 0]:
            torch.cuda.synchronize()  # Ensure GPU operations are complete
            step_end_time = time.time()
            iteration_time = step_end_time - self.step_start_time
            self.iteration_times.append(iteration_time)

        return loss

    def train(self, *args, **kwargs):
        """Override train to print timing statistics after training completes"""
        output = super().train(*args, **kwargs)

        # Print timing statistics on rank 0 only
        if self.args.local_rank in [-1, 0]:
            self._print_timing_statistics()

        return output

    def _print_timing_statistics(self):
        """Calculate and print accurate timing statistics, excluding warmup steps"""
        if len(self.iteration_times) == 0:
            rank0_print("No iteration times recorded.")
            return

        # Get warmup steps from training args
        warmup_steps = self.args.warmup_steps if hasattr(self.args, 'warmup_steps') else 0

        # Ensure we have enough steps
        if len(self.iteration_times) <= warmup_steps:
            rank0_print(f"Warning: Only {len(self.iteration_times)} steps completed, cannot exclude {warmup_steps} warmup steps.")
            warmup_steps = 0

        # Calculate statistics excluding warmup
        if warmup_steps > 0:
            times_after_warmup = self.iteration_times[warmup_steps:]
            avg_time = sum(times_after_warmup) / len(times_after_warmup)
            min_time = min(times_after_warmup)
            max_time = max(times_after_warmup)
        else:
            avg_time = sum(self.iteration_times) / len(self.iteration_times)
            min_time = min(self.iteration_times)
            max_time = max(self.iteration_times)

        # Print results
        rank0_print("\n" + "="*60)
        rank0_print("ACCURATE ITERATION TIMING STATISTICS")
        rank0_print("="*60)
        rank0_print(f"Total steps completed: {len(self.iteration_times)}")
        rank0_print(f"Warmup steps excluded: {warmup_steps}")
        rank0_print(f"Steps used for averaging: {len(self.iteration_times) - warmup_steps}")
        rank0_print("-"*60)
        rank0_print(f"Average iteration time (excluding warmup): {avg_time:.4f} seconds")
        rank0_print(f"Min iteration time: {min_time:.4f} seconds")
        rank0_print(f"Max iteration time: {max_time:.4f} seconds")
        rank0_print(f"Steps per second: {1.0/avg_time:.4f}")
        rank0_print("="*60 + "\n")


class DebugStepsCallback(TrainerCallback):
    """Stop training once a target number of steps is reached."""

    def __init__(self):
        self.triggered = False

    def on_step_end(self, args, state, control, **kwargs):
        target_steps = getattr(args, "debug_steps", None)
        if target_steps is None:
            return control

        if state.global_step >= target_steps:
            if not self.triggered:
                rank0_print(f"Debug steps reached at global step {state.global_step}; stopping training.")
                self.triggered = True
            control.should_training_stop = True
        return control


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen3" in model_args.model_name_or_path.lower() and "moe" in model_args.model_name_or_path.lower():
        if not QWEN3_VL_AVAILABLE:
            raise ImportError(
                "Qwen3VL models are not available in your transformers version. "
                "Please upgrade transformers to use Qwen3VL models."
            )
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_args.model_name_or_path.lower():
        if not QWEN3_VL_AVAILABLE:
            raise ImportError(
                "Qwen3VL models are not available in your transformers version. "
                "Please upgrade transformers to use Qwen3VL models."
            )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        # from transformers import Qwen2_5_VLConfig
        # config = Qwen2_5_VLConfig.from_pretrained(
        #     model_args.model_name_or_path,
        #     cache_dir=training_args.cache_dir,
        #     attn_implementation=attn_implementation,
        # )
        # model = Qwen2_5_VLForConditionalGeneration(config)
        # if training_args.bf16:
        #     model = model.to(torch.bfloat16)
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)

    if torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()
    
    data_module = make_supervised_data_module(processor, data_args=data_args)
    trainer = QwenVLTrainer(
        model=model, 
        processing_class=tokenizer, 
        args=training_args, 
        **data_module
    )

    if training_args.debug_steps is not None:
        rank0_print(f"Debug mode active: training will stop after {training_args.debug_steps} steps.")
        trainer.add_callback(DebugStepsCallback())

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    model.config.use_cache = True

    # Skip saving if save_strategy is "no" (e.g., for benchmarking)
    if training_args.save_strategy != "no":
        trainer.save_state()
        processor.save_pretrained(training_args.output_dir)
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    else:
        rank0_print("Skipping model save (save_strategy='no')")


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
