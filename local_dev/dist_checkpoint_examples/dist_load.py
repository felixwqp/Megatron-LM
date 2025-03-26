
from pathlib import Path

import torch

from megatron.core import dist_checkpointing


def run():
    ckpt_root = Path('/tmp/checkpoints')
    dist_ckpt_root = ckpt_root / 'dist_ckpt'

    torch.distributed.init_process_group()
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    assert 128 % world_size == 0
    num_elems_per_rank = 128 // world_size

    # Local tensor to load
    local_ten = torch.empty(num_elems_per_rank)
    sharded_state_dict = {
        'weight': dist_checkpointing.ShardedTensor.from_rank_offsets('weight', local_ten, (0, rank, world_size))
    }
    loaded_state_dict = dist_checkpointing.load(sharded_state_dict, dist_ckpt_root)
    expected_local_ten = torch.arange(start=num_elems_per_rank * rank, end=num_elems_per_rank * (rank + 1))
    assert torch.all(loaded_state_dict['weight'] == expected_local_ten)

if __name__ == '__main__':
    run()
