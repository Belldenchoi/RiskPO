"""Run the original four-A100 RiskPO notebook workflow from a Python venv."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from config import build_lora_config
from collect_results import collect_results

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MODEL_ID = 'Qwen/Qwen3-4B'
METHOD = 'riskpo'
WORLD_SIZE = 4


def make_config(dataset, model_path, data_dir, checkpoint_dir, run_dir, run_name):
    """Preserve the notebook training settings; change only paths and final output hooks."""
    cfg = build_lora_config(
        dataset, METHOD, MODEL_ID, data_dir, checkpoint_dir, run_name,
        lora_rank=8, lora_alpha=16, train_batch_size=20, ppo_mini_batch_size=20,
        run_profile='full',
    )
    cfg['data']['enable_thinking'] = False
    cfg['data']['custom_cls'] = {'path': str(HERE / 'thinking_dataset.py'), 'name': 'ThinkingModeDataset'}
    cfg['custom_reward_function'] = {'path': str(HERE / 'reward.py'), 'name': 'compute_dataset_score'}
    cfg['actor_rollout_ref']['model']['path'] = str(model_path)
    total = cfg['trainer']['total_training_steps']
    cfg['trainer'].update(
        save_freq=total, test_freq=total, val_before_train=False,
        resume_mode='disable', resume_from_path=None, max_actor_ckpt_to_keep=1,
        validation_data_dir=str(run_dir / 'evaluation'),
    )
    cfg['actor_rollout_ref']['actor']['checkpoint'] = {
        'save_contents': ['model', 'optimizer', 'extra'],
        'load_contents': ['model', 'optimizer', 'extra'],
    }
    return cfg


def run(command, env, **kwargs):
    print('+', ' '.join(str(p) for p in command), flush=True)
    return subprocess.run([str(p) for p in command], cwd=REPO, env=env, check=True, **kwargs)


def resolve_model(model_path, offline):
    if model_path:
        path = Path(model_path).expanduser().resolve()
    else:
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(MODEL_ID, local_files_only=offline)).resolve()
    if not (path / 'config.json').is_file() or not (path / 'tokenizer_config.json').is_file():
        raise RuntimeError(f'Missing model config/tokenizer in {path}')
    index = path / 'model.safetensors.index.json'
    if index.is_file():
        shards = set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values())
    else:
        shards = {'model.safetensors'}
    if not all((path / name).is_file() and (path / name).stat().st_size for name in shards):
        raise RuntimeError(f'Missing full base model weight files in {path}')
    config = json.loads((path / 'config.json').read_text(encoding='utf-8'))
    if config.get('model_type') != 'qwen3' or config.get('quantization_config'):
        raise RuntimeError('Use the original unquantized Qwen/Qwen3-4B checkpoint for this comparison')
    return path


def prepare_data(dataset, data_dir, offline, env):
    raw = data_dir / 'raw'
    if dataset == 'gsm8k':
        command = [sys.executable, HERE / 'prepare_data.py', '--output-dir', raw / 'gsm8k']
        if offline:
            command.append('--offline')
        run(command, env)
        return
    if dataset == 'easymath':
        files = [raw / name / f'{split}.parquet' for name in ('gsm8k', 'math') for split in ('train', 'test')]
        command = [sys.executable, REPO / 'data_processing/download_easymath.py',
                   '--math_local_dir', raw / 'math', '--gsm8k_local_dir', raw / 'gsm8k']
    else:
        files = [raw / 'dapo_aime2024' / name for name in ('dapo-math-17k.parquet', 'aime-2024.parquet')]
        command = [sys.executable, REPO / 'data_processing/download_dapomath.py', '--output_dir', raw]
    missing = [p for p in files if not p.is_file()]
    if missing and offline:
        raise RuntimeError('Offline mode needs preprocessed data: ' + ', '.join(map(str, missing)))
    if missing:
        for path in files:
            path.parent.mkdir(parents=True, exist_ok=True)
        run(command, env)


def train_process(command, env, log_path):
    with log_path.open('w', encoding='utf-8', buffering=1) as log:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
            result = process.wait()
        except KeyboardInterrupt:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
                if process.poll() is not None:
                    break
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    continue
            raise
    if result:
        raise RuntimeError(f'Training failed (exit={result}). See {log_path}; model export was not started.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=REPO / 'riskpo_local_runtime')
    parser.add_argument('--model-path', type=Path, help='Local full Qwen3-4B model; otherwise use Hugging Face cache/download')
    parser.add_argument('--data-dir', type=Path, help='Data root containing raw/gsm8k/{train,test}.parquet')
    parser.add_argument('--dataset', choices=['gsm8k', 'easymath', 'dapo'], default='gsm8k')
    parser.add_argument('--offline', action='store_true', help='Disable Hugging Face downloads; use existing model and processed data')
    parser.add_argument('--prepare-only', action='store_true', help='Check GPU/NCCL, data and config; do not train')
    args = parser.parse_args(argv)
    if not sys.platform.startswith('linux') or sys.version_info[:2] != (3, 10):
        parser.error('Use Linux with the Python 3.10 venv from README.md')
    work = args.work_dir.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir.expanduser().resolve() if args.data_dir else work / 'data/riskpo_reference'
    env = dict(os.environ, VLLM_USE_V1='0', HYDRA_FULL_ERROR='1', TOKENIZERS_PARALLELISM='false',
               PYTHONUNBUFFERED='1', HF_HUB_DISABLE_TELEMETRY='1', RAY_USAGE_STATS_ENABLED='0')
    env.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    env.setdefault('HF_HOME', str(work / 'hf_cache'))
    env['PYTHONPATH'] = str(REPO) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    if args.offline:
        env.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    # Also apply offline/cache settings before importing Hugging Face in this parent process.
    for key in ('HF_HOME', 'HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        if key in env:
            os.environ[key] = env[key]
    run([sys.executable, HERE / 'check_environment.py'], env)
    run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
         '--nproc-per-node=4', HERE / 'distributed_preflight.py'], env, timeout=180)
    base = resolve_model(args.model_path, args.offline)
    prepare_data(args.dataset, data_dir, args.offline, env)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    name = f'local_full_riskpo_Qwen3-4B_{args.dataset}_{stamp}'
    run_dir = work / 'runs' / name
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = work / 'checkpoints' / name
    cfg = make_config(args.dataset, base, data_dir, checkpoint, run_dir, name)
    config_path = REPO / 'verl/trainer/config' / (name + '.yaml')
    config_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    shutil.copy2(config_path, run_dir / 'config.yaml')
    for filename in ('reward.py', 'thinking_dataset.py', 'config.py', 'train.py', 'notebook_provenance.json', 'requirements.txt'):
        shutil.copy2(HERE / filename, run_dir / filename)
    metadata = dict(json.loads((HERE / 'notebook_provenance.json').read_text(encoding='utf-8')),
                    model=MODEL_ID, base_snapshot=str(base), enable_thinking=False, dataset=args.dataset,
                    base_config_sha256=hashlib.sha256((base / 'config.json').read_bytes()).hexdigest(),
                    total_steps=cfg['trainer']['total_training_steps'], offline=args.offline,
                    python=sys.executable, prepare_only=args.prepare_only)
    (run_dir / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], env=env, text=True)
    (run_dir / 'requirements-freeze.txt').write_text(freeze, encoding='utf-8')
    audit = {'config_dir': str(config_path.parent), 'config_name': name, 'run_dir': str(run_dir), 'dataset': args.dataset}
    audit_file = run_dir / 'audit_input.json'
    audit_file.write_text(json.dumps(audit), encoding='utf-8')
    env['TENSORBOARD_DIR'] = str(run_dir / 'tensorboard')
    command = [sys.executable, '-m', 'verl.trainer.main_ppo', '--config-name', name]
    resolved = subprocess.check_output(command + ['--cfg', 'job', '--resolve'], cwd=REPO, env=env, text=True)
    (run_dir / 'resolved_config.yaml').write_text(resolved, encoding='utf-8')
    print(resolved, flush=True)
    run([sys.executable, HERE / 'validate_config.py', audit_file], env)
    run([sys.executable, HERE / 'audit_data.py', audit_file], env)
    print('RUN_DIR:', run_dir, flush=True)
    if args.prepare_only:
        print('Preparation checks passed. No training performed. Run without --prepare-only to train.')
        return
    (run_dir / 'command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
    train_process(command, env, run_dir / 'train.log')
    steps = cfg['trainer']['total_training_steps']
    results = collect_results(checkpoint, run_dir, steps, MODEL_ID, METHOD, args.dataset, WORLD_SIZE)
    final_model = run_dir / 'final_model'
    run([sys.executable, HERE / 'export_model.py', '--base', base, '--adapter', results['adapter'],
         '--output', final_model, '--thinking', 'false', '--step', steps], env)
    manifest = json.loads((final_model / 'export_manifest.json').read_text(encoding='utf-8'))
    if manifest['step'] != steps:
        raise RuntimeError('Exported model step does not match final training step')
    results.update(merged_model_export_complete=True, final_model=str(final_model))
    (run_dir / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('RESULTS:', run_dir / 'results.json')
    print('FULL_MODEL:', final_model)


if __name__ == '__main__':
    main()
