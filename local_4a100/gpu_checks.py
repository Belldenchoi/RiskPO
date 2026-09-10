from pathlib import Path
REQUIRED_GPUS = 4
REWARD_PATH = str(Path(__file__).with_name("reward.py"))

import torch
import transformers
import vllm
import flash_attn
import tensorboard
from verl.trainer.ppo.core_algos import (
    compute_grpo_bundle_rvar_outcome_advantage_quantile_tracking, compute_policy_loss,
)
import importlib.util
spec = importlib.util.spec_from_file_location("riskpo_reward_v2_check", REWARD_PATH)
reward = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reward)
default_compute_score = reward.compute_dataset_score

print("torch", torch.__version__, "CUDA", torch.version.cuda)
print("transformers", transformers.__version__, "vllm", vllm.__version__)
print("flash_attn", flash_attn.__version__)
assert torch.cuda.device_count() == REQUIRED_GPUS == 4, "Expose đúng bốn GPU qua CUDA_VISIBLE_DEVICES."
for i in range(REQUIRED_GPUS):
    print(i, torch.cuda.get_device_name(i))
    assert "A100" in torch.cuda.get_device_name(i), "Cấu hình này dành cho bốn A100."
    assert torch.cuda.get_device_capability(i)[0] >= 8, "Cần GPU hỗ trợ FlashAttention-2/BF16."
assert default_compute_score("openai/gsm8k", "#### 42", "42") == 1
assert default_compute_score("openai/gsm8k", r"\boxed{42}", "42") == 1
assert default_compute_score("openai/gsm8k", r"\boxed{41}", "42") == 0
assert default_compute_score("openai/gsm8k", r"<think>\boxed{42}</think>No final answer.", "42") == 0
assert default_compute_score("openai/gsm8k", r"<think>\boxed{41}</think>Final: \boxed{42}", "42") == 1
print("GSM8K scorer:", reward.GSM8K_SCORER_VERSION)
assert default_compute_score(
    "DigitalLearningGmbH/MATH-lighteval", r"\boxed{42}", "42"
) == 1
print("Import/GPU/reward checks passed; chưa kiểm tra đủ VRAM cho training.")
