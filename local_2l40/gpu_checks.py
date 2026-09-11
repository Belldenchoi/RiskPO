"""Import, CUDA and reward checks for two L40s; does not start training."""
from pathlib import Path
import sys
import torch
import transformers
import vllm
import flash_attn
from verl.trainer.ppo.core_algos import compute_grpo_bundle_rvar_outcome_advantage_quantile_tracking, compute_policy_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'local_4a100'))
from reward import compute_dataset_score

assert torch.cuda.device_count() == 2, 'Expose exactly two GPUs through CUDA_VISIBLE_DEVICES'
print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'CXX11 ABI', torch._C._GLIBCXX_USE_CXX11_ABI)
print('transformers', transformers.__version__, 'vllm', vllm.__version__, 'flash_attn', flash_attn.__version__)
for index in range(2):
    name = torch.cuda.get_device_name(index)
    properties = torch.cuda.get_device_properties(index)
    assert 'L40' in name, f'Expected NVIDIA L40, found {name}'
    assert torch.cuda.get_device_capability(index)[0] >= 8
    with torch.cuda.device(index):
        assert torch.cuda.is_bf16_supported()
        free, total = torch.cuda.mem_get_info()
    print(index, name, 'VRAM GiB', total / 2**30, 'FREE GiB', free / 2**30, flush=True)
    if total < 40 * 2**30:
        raise RuntimeError('This profile expects full L40 VRAM, not a small virtual GPU partition')
assert compute_dataset_score('openai/gsm8k', r'\boxed{42}', '42') == 1
assert compute_dataset_score('openai/gsm8k', '#### 42', '42') == 1
assert compute_dataset_score('openai/gsm8k', r'<think>\boxed{42}</think>No final answer.', '42') == 0
assert compute_dataset_score('DigitalLearningGmbH/MATH-lighteval', r'\boxed{42}', '42') == 1
print('Two-L40 GPU/import/reward checks passed. Actual peak VRAM must be measured during training.')
