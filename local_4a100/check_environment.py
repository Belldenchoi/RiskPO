"""Inspect the target host without installing packages or using network access."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

PINS = {'torch': '2.6.0', 'torchvision': '0.21.0', 'torchaudio': '2.6.0',
        'vllm': '0.8.5.post1', 'transformers': '4.51.3', 'peft': '0.15.2',
        'accelerate': '1.6.0', 'ray': '2.43.0', 'tensordict': '0.8.3',
        'torchdata': '0.11.0', 'datasets': '3.6.0', 'numpy': '1.26.4',
        'hydra-core': '1.3.2', 'flash-attn': '2.7.4.post1'}
REQUIRED = ['tensorboard', 'codetiming', 'dill', 'pandas', 'pyarrow', 'pybind11',
            'pylatexenc', 'wandb', 'packaging', 'math-verify', 'pip', 'setuptools', 'wheel']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path('environment_report.json'))
    parser.add_argument('--metadata-only', action='store_true', help='Skip GPU/library imports and pip check')
    parser.add_argument('--gpu-count', type=int, choices=[2, 4], default=4)
    parser.add_argument('--gpu-family', choices=['A100', 'L40'], default='A100')
    args = parser.parse_args()
    report = {'os': platform.system(), 'architecture': platform.machine(), 'libc': platform.libc_ver(),
              'python': sys.version, 'executable': sys.executable, 'in_venv': sys.prefix != sys.base_prefix,
              'disk_free_gib': round(shutil.disk_usage(Path.cwd()).free / 2**30, 1),
              'packages': {}, 'errors': [], 'checks': {}}
    if report['os'] != 'Linux' or report['architecture'].lower() not in ('x86_64', 'amd64'):
        report['errors'].append('This environment requires Linux x86_64')
    if sys.version_info[:2] != (3, 10):
        report['errors'].append('Use Python 3.10 for the pinned FlashAttention wheel')
    for name in list(PINS) + REQUIRED:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        report['packages'][name] = version
        if version is None:
            report['errors'].append('Missing package: ' + name)
        elif name in PINS and version.split('+')[0] != PINS[name]:
            report['errors'].append(f'{name}: expected {PINS[name]}, found {version}')
    env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
               RAY_USAGE_STATS_ENABLED='0', VLLM_USE_V1='0')
    env.setdefault('CUDA_VISIBLE_DEVICES', ','.join(map(str, range(args.gpu_count))))
    repo = Path(__file__).resolve().parent.parent
    env['PYTHONPATH'] = str(repo) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    commands = {'nvidia_smi': ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']}
    if not args.metadata_only:
        commands['pip_check'] = [sys.executable, '-m', 'pip', 'check']
        if not report['errors']:
            if args.gpu_count == 4 and args.gpu_family == 'A100':
                gpu_check = Path(__file__).with_name('gpu_checks.py')
            elif args.gpu_count == 2 and args.gpu_family == 'L40':
                gpu_check = repo / 'local_2l40/gpu_checks.py'
            else:
                raise ValueError('Supported targets: 4 A100 or 2 L40')
            commands['gpu_import_reward_checks'] = [sys.executable, str(gpu_check)]
    commands['gpu_topology'] = ['nvidia-smi', 'topo', '-m']
    for name, command in commands.items():
        print('Checking:', name, flush=True)
        try:
            result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=120)
            report['checks'][name] = {'exit_code': result.returncode, 'output': result.stdout}
            if result.returncode:
                report['errors'].append(name + ' failed; inspect report output')
        except (OSError, subprocess.TimeoutExpired) as exc:
            report['checks'][name] = {'error': str(exc)}
            report['errors'].append(name + ' could not complete')
    report['status'] = 'failed' if report['errors'] else ('metadata_only' if args.metadata_only else 'passed')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('REPORT:', args.report.resolve())
    print('No installation, downloads or training were performed.')
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
