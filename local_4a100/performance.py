"""Summarize measured step timings; do not infer speedups from GPU counts."""
import json
from pathlib import Path
import re
import statistics

NUMBER = r'[-+]?(?:\d*\.)?\d+(?:[eE][-+]?\d+)?'


def summarize(log_path, expected_steps=None, warmup_steps=1):
    rows = {}
    for line in Path(log_path).read_text(encoding='utf-8', errors='replace').splitlines():
        match = re.search(r'\bstep:(\d+)\s+-', line)
        if not match:
            continue
        values = {key: float(value) for key, value in re.findall(r'([\w/\.]+):(' + NUMBER + r')', line)}
        if 'timing_s/step' in values:
            rows[int(match.group(1))] = values
    if expected_steps is not None and set(rows) != set(range(1, expected_steps + 1)):
        raise RuntimeError(f'Incomplete timing log: got steps {sorted(rows)}, expected 1..{expected_steps}')
    measured = [v for k, v in sorted(rows.items()) if k > warmup_steps]
    if not measured:
        raise RuntimeError('No measured steps after warmup')
    metrics = ('timing_s/step', 'timing_s/gen', 'timing_s/old_log_prob', 'timing_s/update_actor',
               'timing_s/reshard', 'perf/max_memory_allocated_gb', 'perf/max_memory_reserved_gb',
               'response_length/mean')
    report = {'steps': sorted(rows), 'warmup_steps_excluded': warmup_steps, 'measured_steps': len(measured),
              'metrics': {}}
    for key in metrics:
        values = [v[key] for v in measured if key in v]
        if values:
            report['metrics'][key] = {'mean': statistics.mean(values), 'median': statistics.median(values), 'max': max(values)}
    tokens = sum(v.get('perf/total_num_tokens', 0) for v in measured)
    seconds = sum(v['timing_s/step'] for v in measured)
    report['aggregate_tokens_per_second'] = tokens / seconds if seconds > 0 else None
    report['zero_gradient_steps'] = [k for k, v in rows.items() if v.get('actor/grad_norm') == 0]
    report['validation'] = 'Measured timings only; no claim of equivalent samples or post-training accuracy'
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('log', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.log), indent=2))
