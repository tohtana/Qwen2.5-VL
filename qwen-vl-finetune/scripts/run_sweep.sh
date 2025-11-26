#!/bin/bash

# Sweep script to run training with different vision token configurations
# This script will iterate through different max_pixels/min_pixels settings
# and save logs for each run to the logs/ directory
# Each run will execute for a limited number of steps (controlled by max_steps variable)

set -e  # Exit on error (can be commented out if you want to continue after failures)

# Create logs directory if it doesn't exist
LOGS_DIR="./logs"
mkdir -p ${LOGS_DIR}

# Get timestamp for this sweep run
SWEEP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_LOG_DIR="${LOGS_DIR}/sweep_${SWEEP_TIMESTAMP}"
mkdir -p ${SWEEP_LOG_DIR}

echo "=========================================="
echo "Starting training sweep at $(date)"
echo "Logs will be saved to: ${SWEEP_LOG_DIR}"
echo "=========================================="

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
NPROC_PER_NODE=8

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=Qwen/Qwen2.5-VL-32B-Instruct

# Training hyperparameters
lr=2e-7
batch_size=1
grad_accum_steps=1
max_steps=100  # Number of training steps to run for each configuration
warmup_steps=3  # Number of warmup steps

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration
datasets=mscoco2017_val_captions

# Output configuration base
output_dir_base=/mnt/local_storage/qwen-vl-finetune/checkpoints/

# Define all configurations to sweep through
# Format: "name:max_pixels:min_pixels:model_max_length:skip_flag"
# skip_flag: 0=run, 1=skip
CONFIGS=(
    "1k_tokens:200704:200704:8192:0"
    "4k_tokens:802816:802816:8192:0"
    "8k_tokens:1806336:1806336:8192:0"
    "16k_tokens:3211264:3211264:8192:0" # OOM for 32B
    # "32k_tokens:5017600:5017600:16384:0" # Skip for 32B (OOM)
    # "64k_tokens:12845056:12845056:32768:0" # Skip for 32B (OOM)
)

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counter for successful and failed runs
TOTAL_RUNS=0
SUCCESSFUL_RUNS=0
FAILED_RUNS=0
SKIPPED_RUNS=0

# Loop through each configuration
for config in "${CONFIGS[@]}"; do
    # Parse configuration
    IFS=':' read -r name max_pixels min_pixels model_max_length skip_flag <<< "$config"

    # Check if this config should be skipped
    if [ "$skip_flag" -eq 1 ]; then
        echo -e "${YELLOW}[SKIP]${NC} Skipping configuration: ${name} (known to cause OOM)"
        SKIPPED_RUNS=$((SKIPPED_RUNS + 1))
        continue
    fi

    TOTAL_RUNS=$((TOTAL_RUNS + 1))

    echo ""
    echo "=========================================="
    echo -e "${GREEN}[RUN ${TOTAL_RUNS}]${NC} Starting configuration: ${name}"
    echo "  max_pixels: ${max_pixels}"
    echo "  min_pixels: ${min_pixels}"
    echo "  model_max_length: ${model_max_length}"
    echo "  max_steps: ${max_steps}"
    echo "  warmup_steps: ${warmup_steps}"
    echo "  Started at: $(date)"
    echo "=========================================="

    # Set unique run name and output directory for this configuration
    run_name="qwen2vl-coco-${name}"
    output_dir="${output_dir_base}${name}_${SWEEP_TIMESTAMP}"

    # Generate unique port for this run
    MASTER_PORT=$(shuf -i 20001-29999 -n 1)

    # Log file for this run
    log_file="${SWEEP_LOG_DIR}/${name}.log"

    # Build training arguments
    args="
        --deepspeed ${deepspeed} \
        --model_name_or_path ${llm} \
        --dataset_use ${datasets} \
        --data_flatten True \
        --tune_mm_vision True \
        --tune_mm_mlp True \
        --tune_mm_llm True \
        --bf16 \
        --output_dir ${output_dir} \
        --max_steps ${max_steps} \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size $((batch_size*2)) \
        --gradient_accumulation_steps ${grad_accum_steps} \
        --max_pixels ${max_pixels} \
        --min_pixels ${min_pixels} \
        --force_fixed_size True \
        --model_max_length ${model_max_length} \
        --eval_strategy no \
        --save_strategy no \
        --learning_rate ${lr} \
        --weight_decay 0 \
        --warmup_steps ${warmup_steps} \
        --max_grad_norm 1 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --run_name ${run_name} \
        --report_to none"

    # Write configuration info to log file
    echo "Configuration: ${name}" > ${log_file}
    echo "Started at: $(date)" >> ${log_file}
    echo "max_pixels: ${max_pixels}" >> ${log_file}
    echo "min_pixels: ${min_pixels}" >> ${log_file}
    echo "model_max_length: ${model_max_length}" >> ${log_file}
    echo "max_steps: ${max_steps}" >> ${log_file}
    echo "warmup_steps: ${warmup_steps}" >> ${log_file}
    echo "output_dir: ${output_dir}" >> ${log_file}
    echo "========================================" >> ${log_file}
    echo "" >> ${log_file}

    # Launch training and capture both stdout and stderr
    set +e  # Temporarily disable exit on error to capture failures
    torchrun --nproc_per_node=${NPROC_PER_NODE} \
             --master_addr=${MASTER_ADDR} \
             --master_port=${MASTER_PORT} \
             ${entry_file} ${args} >> ${log_file} 2>&1

    exit_code=$?
    set -e  # Re-enable exit on error

    # Extract accurate iteration timing from the training script output
    avg_iter_time="N/A"
    train_steps_per_sec="N/A"
    min_iter_time="N/A"
    max_iter_time="N/A"

    # Look for the accurate timing statistics printed by QwenVLTrainer
    if grep -q "ACCURATE ITERATION TIMING STATISTICS" ${log_file}; then
        # Extract average iteration time (excluding warmup)
        avg_iter_time=$(grep "Average iteration time (excluding warmup):" ${log_file} | tail -1 | grep -oP '\K[0-9.]+(?= seconds)')

        # Extract steps per second
        train_steps_per_sec=$(grep "Steps per second:" ${log_file} | tail -1 | grep -oP '\K[0-9.]+$')

        # Extract min and max times
        min_iter_time=$(grep "Min iteration time:" ${log_file} | tail -1 | grep -oP '\K[0-9.]+(?= seconds)')
        max_iter_time=$(grep "Max iteration time:" ${log_file} | tail -1 | grep -oP '\K[0-9.]+(?= seconds)')
    fi

    # Record completion time and status
    echo "" >> ${log_file}
    echo "========================================" >> ${log_file}
    echo "Completed at: $(date)" >> ${log_file}
    echo "Exit code: ${exit_code}" >> ${log_file}
    echo "Extracted Timing Metrics:" >> ${log_file}
    echo "  Average iteration time: ${avg_iter_time}s" >> ${log_file}
    echo "  Steps per second: ${train_steps_per_sec}" >> ${log_file}
    echo "  Min iteration time: ${min_iter_time}s" >> ${log_file}
    echo "  Max iteration time: ${max_iter_time}s" >> ${log_file}

    # Check if training was successful
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}[SUCCESS]${NC} Configuration ${name} completed successfully"
        echo "  Average iteration time (excl. warmup): ${avg_iter_time}s"
        echo "  Steps per second: ${train_steps_per_sec}"
        echo "Status: SUCCESS" >> ${log_file}
        SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
    else
        echo -e "${RED}[FAILED]${NC} Configuration ${name} failed with exit code ${exit_code}"
        echo "Status: FAILED" >> ${log_file}
        FAILED_RUNS=$((FAILED_RUNS + 1))

        # Optionally, uncomment the next line to stop the sweep on first failure
        # exit 1
    fi

    echo "Log saved to: ${log_file}"

    # Optional: Add a delay between runs to allow system to cool down
    # sleep 60
done

# Print summary
echo ""
echo "=========================================="
echo "Sweep completed at $(date)"
echo "=========================================="
echo "Summary:"
echo "  Total configurations: ${#CONFIGS[@]}"
echo "  Executed: ${TOTAL_RUNS}"
echo -e "  ${GREEN}Successful: ${SUCCESSFUL_RUNS}${NC}"
echo -e "  ${RED}Failed: ${FAILED_RUNS}${NC}"
echo -e "  ${YELLOW}Skipped: ${SKIPPED_RUNS}${NC}"
echo "=========================================="
echo ""
echo "Timing Results (excluding warmup steps):"
echo "----------------------------------------"
printf "%-15s | %-10s | %-20s | %-15s\n" "Configuration" "Status" "Avg Iter Time (s)" "Steps/Sec"
echo "----------------------------------------"
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r name max_pixels min_pixels model_max_length skip_flag <<< "$config"
    if [ "$skip_flag" -eq 1 ]; then
        printf "%-15s | %-10s | %-20s | %-15s\n" "$name" "SKIPPED" "N/A" "N/A"
    elif [ -f "${SWEEP_LOG_DIR}/${name}.log" ]; then
        # Extract accurate timing info from log (from Extracted Timing Metrics section)
        avg_time=$(grep "Average iteration time:" "${SWEEP_LOG_DIR}/${name}.log" | tail -1 | grep -oP '\K[0-9.]+(?=s)' || echo "N/A")
        steps_per_sec=$(grep "Steps per second:" "${SWEEP_LOG_DIR}/${name}.log" | tail -1 | awk '{print $NF}')

        if grep -q "Status: SUCCESS" "${SWEEP_LOG_DIR}/${name}.log"; then
            printf "%-15s | ${GREEN}%-10s${NC} | %-20s | %-15s\n" "$name" "SUCCESS" "${avg_time}" "$steps_per_sec"
        else
            printf "%-15s | ${RED}%-10s${NC} | %-20s | %-15s\n" "$name" "FAILED" "${avg_time}" "$steps_per_sec"
        fi
    fi
done
echo "=========================================="
echo "All logs saved to: ${SWEEP_LOG_DIR}"

# Create summary file
summary_file="${SWEEP_LOG_DIR}/summary.txt"
echo "Training Sweep Summary" > ${summary_file}
echo "======================" >> ${summary_file}
echo "Started: ${SWEEP_TIMESTAMP}" >> ${summary_file}
echo "Completed: $(date)" >> ${summary_file}
echo "" >> ${summary_file}
echo "Total configurations: ${#CONFIGS[@]}" >> ${summary_file}
echo "Executed: ${TOTAL_RUNS}" >> ${summary_file}
echo "Successful: ${SUCCESSFUL_RUNS}" >> ${summary_file}
echo "Failed: ${FAILED_RUNS}" >> ${summary_file}
echo "Skipped: ${SKIPPED_RUNS}" >> ${summary_file}
echo "" >> ${summary_file}
echo "Configurations:" >> ${summary_file}
echo "Format: name | status | avg_iter_time (excl. warmup) | steps_per_sec" >> ${summary_file}
echo "--------------------------------------------------------------" >> ${summary_file}
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r name max_pixels min_pixels model_max_length skip_flag <<< "$config"
    if [ "$skip_flag" -eq 1 ]; then
        echo "  ${name}: SKIPPED (OOM)" >> ${summary_file}
    elif [ -f "${SWEEP_LOG_DIR}/${name}.log" ]; then
        # Extract accurate timing info from log (from Extracted Timing Metrics section)
        avg_time=$(grep "Average iteration time:" "${SWEEP_LOG_DIR}/${name}.log" | tail -1 | grep -oP '\K[0-9.]+(?=s)' || echo "N/A")
        steps_per_sec=$(grep "Steps per second:" "${SWEEP_LOG_DIR}/${name}.log" | tail -1 | awk '{print $NF}')

        if grep -q "Status: SUCCESS" "${SWEEP_LOG_DIR}/${name}.log"; then
            echo "  ${name}: SUCCESS | ${avg_time}s | ${steps_per_sec} steps/s" >> ${summary_file}
        else
            echo "  ${name}: FAILED | ${avg_time}s | ${steps_per_sec} steps/s" >> ${summary_file}
        fi
    fi
done

echo ""
echo "Summary saved to: ${summary_file}"

# Exit with error if any runs failed (optional)
if [ $FAILED_RUNS -gt 0 ]; then
    exit 1
fi
