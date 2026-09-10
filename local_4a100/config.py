from pathlib import Path

def build_reference_config(dataset, method, model_id, data_root, checkpoint_dir, run_name):
    """Match executable overrides in the RiskPO scripts, then apply the chosen method."""
    if dataset not in {"gsm8k", "easymath", "dapo"} or method != "riskpo":
        raise ValueError("Unknown dataset or method")
    hard = dataset == "dapo"
    raw = Path(data_root) / "raw"
    train_files = ([str(raw / "dapo_aime2024/dapo-math-17k.parquet")] if hard else
                   [str(raw / "gsm8k/train.parquet"), str(raw / "math/train.parquet")])
    val_files = ([str(raw / "dapo_aime2024/aime-2024.parquet")] if hard else
                 [str(raw / "gsm8k/test.parquet"), str(raw / "math/test.parquet")])
    if dataset == "gsm8k":
        train_files = [str(raw / "gsm8k/train.parquet")]
        val_files = [str(raw / "gsm8k/test.parquet")]
    algorithm = {
        "adv_estimator": "grpo_bundle_RVaR_quantile_tracking",
        "quantile_down": 0.2, "quantile_up": 0.8 if hard else 0.9,
        "bundle_size": 5, "lr_q": 0.1,
        "credit_assign_mode": "sum-mean" if hard else "std",
        "use_q_track_mode": "track", "norm_adv_by_std_in_grpo": True,
        "use_mixing_risk_measure": True, "w_mix": 1.5,
        "use_mean_as_baseline": False, "natural_baseline_base": True,
        "natural_baseline_adv": not hard,
        "quantile_tracking": True, "use_kl_in_reward": False,
    }
    actor = {
        "optim": {"lr": 1e-6},
        "ppo_mini_batch_size": 128 if hard else 512,
        "ppo_micro_batch_size_per_gpu": 64,
        "use_kl_loss": False, "kl_loss_coef": 0.001,
        "kl_loss_type": "low_var_kl", "entropy_coeff": 0,
        "fsdp_config": {"param_offload": hard, "optimizer_offload": False},
        "policy_loss": {"loss_mode": "vanilla"},
        "loss_agg_mode": "token-mean",
    }
    rollout = {
        "name": "vllm", "mode": "sync",
        "log_prob_micro_batch_size_per_gpu": 16 if hard else 64,
        "tensor_model_parallel_size": 2, "gpu_memory_utilization": 0.8,
        "n": 10 if hard else 5,
        "temperature": 1.0, "top_p": 1.0, "top_k": -1,
        "val_kwargs": {"do_sample": False, "n": 1,
                       "temperature": 0.0, "top_p": 1.0, "top_k": -1},
    }
    ref = {
        "log_prob_micro_batch_size_per_gpu": 16 if hard else 64,
        "fsdp_config": {"param_offload": True},
    }
    if hard:
        actor.update(use_dynamic_bsz=False, ppo_max_token_len_per_gpu=4096,
                     ulysses_sequence_parallel_size=4)
        ref.update(log_prob_use_dynamic_bsz=False, log_prob_max_token_len_per_gpu=4096,
                   ulysses_sequence_parallel_size=4)
        rollout.update(log_prob_use_dynamic_bsz=False,
                       log_prob_max_token_len_per_gpu=4096, max_num_batched_tokens=4096)
    return {
        "defaults": ["ppo_trainer", "_self_"],
        "algorithm": algorithm,
        "data": {
            "train_files": train_files, "val_files": val_files,
            "train_batch_size": 512 if hard else 1024,
            "max_prompt_length": 1024, "max_response_length": 3072 if hard else 1024,
            "filter_overlong_prompts": True, "truncation": "error",
        },
        "actor_rollout_ref": {
            "model": {"path": model_id, "lora_rank": 0,
                      "use_remove_padding": True, "enable_gradient_checkpointing": True},
            "actor": actor, "rollout": rollout, "ref": ref,
        },
        "trainer": {
            "critic_warmup": 0, "logger": ["console", "tensorboard"],
            "project_name": "riskpo_reference_" + dataset, "experiment_name": run_name,
            "n_gpus_per_node": 8 if hard else 4, "nnodes": 1,
            "save_freq": 10, "test_freq": 5,
            "total_training_steps": 500 if hard else 200,
            "total_epochs": 30 if hard else 15, "val_before_train": True,
            "default_local_dir": str(checkpoint_dir),
        },
    }

def build_lora_config(dataset, method, model_id, data_root, checkpoint_dir, run_name,
                      lora_rank=8, lora_alpha=16, train_batch_size=20, ppo_mini_batch_size=20,
                      run_profile="full"):
    """Keep original data splits; adapt training to four-GPU BF16 LoRA."""
    if run_profile not in {"smoke", "full"}:
        raise ValueError("run_profile must be smoke or full")
    if lora_rank not in {8, 16, 32, 64} or lora_alpha <= 0:
        raise ValueError("Use a supported LoRA rank (8/16/32/64) and positive alpha")
    if (train_batch_size <= 0 or ppo_mini_batch_size <= 0
            or train_batch_size % ppo_mini_batch_size != 0
            or train_batch_size % 5 != 0):
        raise ValueError("Batch must contain full bundles of 5 and be divisible by PPO mini-batch")
    cfg = build_reference_config(dataset, method, model_id, data_root, checkpoint_dir, run_name)
    cfg["data"].update(train_batch_size=train_batch_size, val_batch_size=4,
                       dataloader_num_workers=2)
    model = cfg["actor_rollout_ref"]["model"]
    model.update(lora_rank=lora_rank, lora_alpha=lora_alpha,
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"])
    actor = cfg["actor_rollout_ref"]["actor"]
    actor.update(ppo_mini_batch_size=ppo_mini_batch_size,
                 ppo_micro_batch_size_per_gpu=1, use_dynamic_bsz=False,
                 use_torch_compile=False, ulysses_sequence_parallel_size=1)
    actor["fsdp_config"].update(model_dtype="bf16", param_offload=True, optimizer_offload=True, fsdp_size=4)
    rollout = cfg["actor_rollout_ref"]["rollout"]
    rollout.update(
        tensor_model_parallel_size=1, dtype="bfloat16",
        log_prob_micro_batch_size_per_gpu=1, log_prob_use_dynamic_bsz=False,
        load_format="safetensors", layered_summon=True,
        gpu_memory_utilization=0.6, enforce_eager=True, free_cache_engine=True,
        max_num_seqs=24,
        max_num_batched_tokens=max(1024, cfg["data"]["max_prompt_length"] + cfg["data"]["max_response_length"]),
        engine_kwargs={"vllm": {"max_num_seqs": 24, "swap_space": 2}},
    )
    cfg["actor_rollout_ref"]["ref"].update(
        log_prob_micro_batch_size_per_gpu=1, log_prob_use_dynamic_bsz=False,
        ulysses_sequence_parallel_size=1)
    cfg["trainer"].update(n_gpus_per_node=4, nnodes=1, max_actor_ckpt_to_keep=1,
                          resume_mode="disable", val_before_train=False, test_freq=-1)
    if run_profile == "smoke":
        cfg["trainer"].update(total_training_steps=5, save_freq=5)
    n = cfg["actor_rollout_ref"]["rollout"]["n"]
    if (train_batch_size * n) % 4 or (ppo_mini_batch_size * n) % 4:
        raise ValueError("Global response batch and PPO minibatch must be divisible by four ranks")
    return cfg
