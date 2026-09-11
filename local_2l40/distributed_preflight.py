"""Check actual NCCL communication between the two allocated GPUs."""
import os
from datetime import timedelta
import torch
import torch.distributed as dist

rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl', timeout=timedelta(seconds=120))
try:
    assert dist.get_world_size() == 2
    value = torch.tensor([float(dist.get_rank() + 1)], device=f'cuda:{rank}')
    dist.all_reduce(value)
    torch.cuda.synchronize()
    assert value.item() == 3.0
    print(f'NCCL_OK rank={dist.get_rank()} GPU={torch.cuda.get_device_name(rank)} sum={value.item()}', flush=True)
finally:
    dist.destroy_process_group()
