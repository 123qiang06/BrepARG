#!/usr/bin/env python3
import os
import torch
import torch.distributed as dist
import traceback
from utils import get_ar_args
from trainer import ARTrainer
from dataset import ARData

def setup_distributed():
    """
    Check distributed environment variables. If present, initialize the process group.
    Returns (local_rank, world_size, rank); returns (None, None, None) if not in distributed mode.
    """
    if 'RANK' not in os.environ:
        return None, None, None  # Determine as non-DDP mode

    # Get distributed training parameters from environment variables
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])

    # Set the corresponding GPU device for the current process
    torch.cuda.set_device(local_rank)
    
    # Initialize the process group; 'nccl' is the recommended backend for NVIDIA GPUs
    dist.init_process_group(backend='nccl', init_method='env://')

    return local_rank, world_size, rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def single_gpu_train(args):
    """
    Full training pipeline for single-GPU mode.
    Defaults to cuda:0, or the first device specified by CUDA_VISIBLE_DEVICES.
    """
    print("\n[Mode] Single-GPU Training")
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        device = torch.device('cpu')
        print("[Warning] No CUDA device detected; training will proceed on CPU.")

    train_dataset = ARData(sequence_file=args.sequence_file, validate=False, args=args)
    val_dataset = ARData(sequence_file=args.sequence_file, validate=True, args=args)

    trainer = ARTrainer(train_dataset, val_dataset, args, device=device, multi_gpu=False)

    trainer.train()


def multi_gpu_train(args, local_rank, world_size, rank):
    """
    Full training pipeline for DDP multi-GPU mode.
    This function is executed separately by each process launched via torchrun.
    """
    device = torch.device(f'cuda:{local_rank}')

    # Print information only on the main process (rank=0) to avoid log clutter
    if rank == 0:
        print("\n[Mode] DDP Multi-GPU Distributed Training")
        print(f"Cluster info: World Size = {world_size} processes")

    train_dataset = ARData(sequence_file=args.sequence_file, validate=False, args=args)
    val_dataset = ARData(sequence_file=args.sequence_file, validate=True, args=args)
    # ARTrainer should internally handle DDP model wrapping
    trainer = ARTrainer(train_dataset, val_dataset, args, device=device, multi_gpu=True)
    trainer.train()

def main():
    try:
        # Parse command-line arguments
        args = get_ar_args()

        # Check if running in DDP environment
        local_rank, world_size, rank = setup_distributed()

        if local_rank is None:
            # If not in DDP mode, run single-GPU training
            single_gpu_train(args)
        else:
            # If in DDP mode, run multi-GPU training
            multi_gpu_train(args, local_rank, world_size, rank)

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"An unhandled exception occurred during training: {e}")
        traceback.print_exc()
    finally:
        # Ensure distributed environment is cleaned up before exit to avoid zombie processes
        cleanup_distributed()


if __name__ == "__main__":
    main()