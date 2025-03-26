from pathlib import Path

import torch
import shutil
import logging

from megatron.core import dist_checkpointing

from local_dev.utils import env_setup



def run():
    env_setup.set_print_env()

    # Setup

    ckpt_root = Path('/tmp/checkpoints')

    if Path(ckpt_root).exists():
        shutil.rmtree(ckpt_root, ignore_errors=True)  # Deletes the entire directory and its contents

    native_ckpt_root = ckpt_root / 'native'
    native_ckpt_root.mkdir(exist_ok=True, parents=True)
    dist_ckpt_root = ckpt_root / 'dist_ckpt'
    dist_ckpt_root.mkdir(exist_ok=True, parents=True)

    # # Torch setup for distributed training
    # rank = int(os.environ['LOCAL_RANK'])
    # world_size = torch.cuda.device_count()


    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)


    print(world_size, rank)

    # Local tensor to save
    assert 128 % world_size == 0
    num_elems_per_rank = 128 // world_size
    local_ten = torch.arange(start=num_elems_per_rank * rank,
                            end=num_elems_per_rank * (rank + 1))

    # Native checkpoint save
    state_dict = {
        'weight': local_ten
    }
    torch.save(state_dict, native_ckpt_root / f'ckpt_{rank}.pt')

    print("Saved native checkpoint")

    # Distributed checkpoint save
    # `(0, rank, world_size)` describes that `weight` ShardedTensor is sharded into `world_size` pieces
    # along the 0th dimension and `local_ten` is the shard at position `rank`.
    # Together, all shards implicitly form a "global" `torch.arange(128)` tensor.
    sharded_state_dict = {
        'weight': dist_checkpointing.ShardedTensor.from_rank_offsets('weight', local_ten, (0, rank, world_size))
    }
    print(sharded_state_dict)
    dist_checkpointing.save(sharded_state_dict, dist_ckpt_root)
    print("Saved dist checkpoint")


if __name__ == '__main__':
    run()