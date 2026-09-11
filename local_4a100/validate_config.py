import argparse
import json
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("audit_file", type=Path)
args = parser.parse_args()
AUDIT = json.loads(args.audit_file.read_text(encoding="utf-8"))
WORLD = AUDIT.get('world_size', 4)
METHOD = "riskpo"

from hydra import initialize_config_dir, compose
import torch
with initialize_config_dir(config_dir=AUDIT["config_dir"], version_base=None):
    cfg = compose(config_name=AUDIT["config_name"])
assert torch.cuda.device_count() == cfg.trainer.n_gpus_per_node == WORLD
assert cfg.trainer.nnodes == 1
if AUDIT.get('benchmark_steps'):
    assert cfg.trainer.total_training_steps == AUDIT['benchmark_steps']
    assert cfg.trainer.save_freq == cfg.trainer.test_freq == -1
else:
    assert cfg.trainer.total_training_steps in (200, 500)
    assert cfg.trainer.save_freq == cfg.trainer.test_freq == cfg.trainer.total_training_steps
assert not cfg.trainer.val_before_train
assert cfg.trainer.resume_mode == "disable"
assert not cfg.trainer.get("durable_checkpoint")
assert cfg.actor_rollout_ref.model.lora_rank > 0
assert cfg.actor_rollout_ref.actor.fsdp_config.model_dtype == "bf16"
assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 1
assert cfg.actor_rollout_ref.actor.fsdp_config.fsdp_size == WORLD
assert cfg.actor_rollout_ref.rollout.tensor_model_parallel_size == 1
assert cfg.data.train_batch_size * cfg.actor_rollout_ref.rollout.n % WORLD == 0
assert not cfg.actor_rollout_ref.rollout.val_kwargs.do_sample
assert cfg.actor_rollout_ref.rollout.val_kwargs.n == 1
if METHOD == "riskpo":
    assert cfg.algorithm.adv_estimator == "grpo_bundle_RVaR_quantile_tracking"
    assert cfg.algorithm.quantile_tracking
    assert cfg.actor_rollout_ref.actor.policy_loss.loss_mode == "vanilla"
else:
    assert cfg.algorithm.adv_estimator == "risk_quatro"
    assert not cfg.algorithm.quantile_tracking
    assert cfg.actor_rollout_ref.actor.policy_loss.loss_mode == "risk_quatro"
    assert cfg.actor_rollout_ref.actor.policy_loss.quatro_ratio_mode == "geometric"
print("Local full-run config checks passed; training not started yet.")
