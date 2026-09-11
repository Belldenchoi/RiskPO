"""Execution-only settings. Research hyperparameters stay in config.py."""
PROFILES = ('a100_4', 'l40_2_fast', 'l40_2_safe', 'l40_2_graph')


def hardware(profile):
    if profile not in PROFILES:
        raise ValueError(f'Unknown hardware profile: {profile}')
    return (4, 'A100') if profile == 'a100_4' else (2, 'L40')


def apply_hardware(cfg, profile):
    world, _ = hardware(profile)
    if profile == 'a100_4':
        return cfg
    actor = cfg['actor_rollout_ref']['actor']
    rollout = cfg['actor_rollout_ref']['rollout']
    cfg['trainer']['n_gpus_per_node'] = world
    actor['fsdp_config']['fsdp_size'] = world
    if profile != 'l40_2_safe':
        # Avoid CPU transfer between rollout / old log-prob / actor updates.
        actor['fsdp_config'].update(param_offload=False, optimizer_offload=False)
        # No-gradient log-prob batching does not change the actor loss reduction.
        rollout['log_prob_micro_batch_size_per_gpu'] = 5
        # Actual per-engine concurrency: the engine consumes engine_kwargs.vllm.
        per_rank_responses = cfg['data']['train_batch_size'] * rollout['n'] // world
        rollout['max_num_seqs'] = per_rank_responses
        rollout['engine_kwargs']['vllm']['max_num_seqs'] = per_rank_responses
        rollout['max_num_batched_tokens'] = max(
            4096, cfg['data']['max_prompt_length'] + cfg['data']['max_response_length'])
        if profile == 'l40_2_graph':
            # Opt-in: captures add startup time/VRAM; benchmark LoRA rollout first.
            rollout['enforce_eager'] = False
    responses = cfg['data']['train_batch_size'] * rollout['n']
    minibatch_responses = actor['ppo_mini_batch_size'] * rollout['n']
    if responses % world or minibatch_responses % world:
        raise ValueError('Response batch must divide evenly across GPU ranks')
    if actor['ppo_micro_batch_size_per_gpu'] != 1 or actor['use_dynamic_bsz']:
        raise ValueError('Keep actor microbatch=1 and static batching for notebook loss weighting')
    return cfg
