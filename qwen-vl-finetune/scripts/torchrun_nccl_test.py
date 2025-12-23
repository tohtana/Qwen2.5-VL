#!/usr/bin/env python3
"""
Simple NCCL communication test for torchrun multi-node setup.

Usage (2 nodes, 8 GPUs each):
    # On node0:
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
        --master_addr=10.1.4.115 --master_port=29500 \
        scripts/torchrun_nccl_test.py

    # On node1:
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
        --master_addr=10.1.4.115 --master_port=29500 \
        scripts/torchrun_nccl_test.py
"""

import os
import time
import torch
import torch.distributed as dist


def main():
    # Get distributed info
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"[Rank {rank}/{world_size}] Starting on GPU {local_rank}")
    print(f"[Rank {rank}] NCCL_NET_PLUGIN={os.environ.get('NCCL_NET_PLUGIN', 'not set')}")

    # Initialize distributed
    if not dist.is_initialized():
        print(f"[Rank {rank}] Initializing process group...")
        dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Test tensor (256 MB)
    tensor_size = 256 * 1024 * 1024 // 4  # 4 bytes per float32
    test_tensor = torch.ones(tensor_size, device=device) * (rank + 1)

    print(f"[Rank {rank}] Created test tensor of size {tensor_size * 4 / 1024 / 1024:.2f} MB")

    # Warmup
    num_warmup = 5
    for i in range(num_warmup):
        dist.all_reduce(test_tensor)
        torch.cuda.synchronize()

    print(f"[Rank {rank}] Warmup complete ({num_warmup} iterations)")

    # Benchmark
    num_iterations = 20
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

    # Calculate bandwidth
    data_size = tensor_size * 4
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
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
