#!/usr/bin/env python3
"""
Simple Ray-based NCCL communication benchmark to test EFA multi-node connectivity.

Usage:
    ray start --head --port 6379  # on node0
    ray start --address='10.1.4.115:6379'  # on node1
    python scripts/ray_nccl_test.py --num_workers 16
"""

import argparse
import os
import time

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

import ray
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer


def prepare_efa_environment():
    """Set EFA environment variables like the working multimodal-training code."""
    env_vars = {
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
        "NCCL_NET_PLUGIN": "ofi",
        "LD_LIBRARY_PATH": "/opt/aws-ofi-nccl/lib:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/usr/local/cuda/lib64",
        "FI_EFA_USE_DEVICE_RDMA": "1",
        "NCCL_DEBUG": "INFO",
    }
    return env_vars


def train_loop(config):
    """Training loop that tests NCCL all-reduce communication."""
    import torch
    import torch.distributed as dist

    # Get distributed info from Ray
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"[Rank {rank}/{world_size}] Starting on GPU {local_rank}")

    # Initialize distributed if not already done
    if not dist.is_initialized():
        print(f"[Rank {rank}] Initializing distributed process group")
        dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Test tensor
    tensor_size = config.get("tensor_size_mb", 1) * 1024 * 1024 // 4  # 4 bytes per float32
    test_tensor = torch.ones(tensor_size, device=device) * (rank + 1)

    print(f"[Rank {rank}] Created test tensor of size {tensor_size * 4 / 1024 / 1024:.2f} MB")

    # Warmup
    num_warmup = config.get("num_warmup", 5)
    for i in range(num_warmup):
        dist.all_reduce(test_tensor)
        torch.cuda.synchronize()

    print(f"[Rank {rank}] Warmup complete ({num_warmup} iterations)")

    # Benchmark
    num_iterations = config.get("num_iterations", 20)
    times = []

    dist.barrier()

    for i in range(num_iterations):
        test_tensor.fill_(rank + 1)
        torch.cuda.synchronize()

        start = time.perf_counter()
        dist.all_reduce(test_tensor)
        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append(end - start)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    # Calculate bandwidth (all-reduce sends 2*size data total)
    data_size = tensor_size * 4  # bytes
    bandwidth_gbps = (2 * data_size) / avg_time / 1e9

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"NCCL All-Reduce Benchmark Results (world_size={world_size})")
        print(f"{'='*60}")
        print(f"Data size: {data_size / 1024 / 1024:.2f} MB")
        print(f"Iterations: {num_iterations}")
        print(f"Average time: {avg_time * 1000:.3f} ms")
        print(f"Min time: {min_time * 1000:.3f} ms")
        print(f"Max time: {max_time * 1000:.3f} ms")
        print(f"Bandwidth: {bandwidth_gbps:.2f} GB/s")
        print(f"{'='*60}\n")

    dist.barrier()

    return {"avg_time_ms": avg_time * 1000, "bandwidth_gbps": bandwidth_gbps}


def main():
    parser = argparse.ArgumentParser(description="Ray-based NCCL communication benchmark")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers (GPUs)")
    parser.add_argument("--tensor_size_mb", type=int, default=256, help="Size of test tensor in MB")
    parser.add_argument("--num_iterations", type=int, default=20, help="Number of benchmark iterations")
    parser.add_argument("--num_warmup", type=int, default=5, help="Number of warmup iterations")
    args = parser.parse_args()

    # Get EFA environment variables
    env_vars = prepare_efa_environment()

    # Initialize Ray with environment variables
    if not ray.is_initialized():
        print(f"Initializing Ray with EFA environment variables:")
        for k, v in env_vars.items():
            print(f"  {k}={v}")
        ray.init(runtime_env={"env_vars": env_vars})

    # Scaling config
    scaling_config = ScalingConfig(
        num_workers=args.num_workers,
        use_gpu=True,
    )

    # Run config
    run_config = RunConfig(
        storage_path="/tmp/ray_nccl_test",
        name="nccl_benchmark",
    )

    # Training config
    train_config = {
        "tensor_size_mb": args.tensor_size_mb,
        "num_iterations": args.num_iterations,
        "num_warmup": args.num_warmup,
    }

    # Create and run trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    print(f"\nStarting NCCL benchmark with {args.num_workers} workers")
    print(f"Tensor size: {args.tensor_size_mb} MB")
    print(f"Iterations: {args.num_iterations}")

    result = trainer.fit()

    print(f"\nBenchmark complete!")


if __name__ == "__main__":
    main()
