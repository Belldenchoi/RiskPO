import ast
import contextlib
import datetime
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import train
from collect_results import collect_results


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = json.loads((REPO / 'RiskPO_Original_Qwen3_4B_4xA100_Local_Full.ipynb').read_text(encoding='utf-8'))

    def cell(self, needle):
        return next(''.join(c['source']) for c in self.nb['cells']
                    if c['cell_type'] == 'code' and needle in ''.join(c['source']))

    def test_all_dataset_configs_equal_notebook(self):
        for dataset in ('gsm8k', 'easymath', 'dapo'):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as temporary:
                p = Path(temporary)
                repo = p / 'repo'
                (repo / 'verl/trainer/config').mkdir(parents=True)
                reward = p / 'reward.py'
                reward.write_text('# mock artifact for config construction\n')
                ctx = dict(Path=Path, datetime=datetime, json=json, shutil=shutil,
                           METHOD='riskpo', MODEL_ID='Qwen/Qwen3-4B', DATASET=dataset,
                           OUTPUT_ROOT=p, CHECKPOINT_ROOT=p / 'checkpoints', DATA_DIR=p / 'data',
                           LORA_RANK=8, LORA_ALPHA=16, TRAIN_BATCH_SIZE=20, PPO_MINI_BATCH_SIZE=20,
                           ENABLE_THINKING=False, GSM8K_REWARD_PATH=reward, REQUIRED_GPUS=4,
                           BASE_SNAPSHOT={'path': str(p / 'base'), 'revision': 'test'}, REPO=repo,
                           ENV={}, UV='uv', PY='python', subprocess=SimpleNamespace(check_output=lambda *a, **k: 'test'))
                exec(self.cell('def build_reference_config('), ctx)
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(self.cell('THINKING_DATASET_SOURCE ='), ctx)
                actual = train.make_config(dataset, p / 'base', ctx['DATA_DIR'], ctx['CKPT_DIR'], ctx['RUN_DIR'], ctx['RUN_NAME'])
                expected = ctx['CONFIG']
                # Notebook writes these helpers into runtime folders; Python uses packaged files.
                expected['data']['custom_cls']['path'] = str(HERE / 'thinking_dataset.py')
                expected['custom_reward_function']['path'] = str(HERE / 'reward.py')
                self.assertEqual(actual, expected)

    def test_extracted_research_code_unchanged(self):
        for name, variable in [('reward.py', 'GSM8K_REWARD_SOURCE'),
                               ('thinking_dataset.py', 'THINKING_DATASET_SOURCE'),
                               ('distributed_preflight.py', 'DISTRIBUTED_PREFLIGHT_SOURCE'),
                               ('export_model.py', 'EXPORT_MODEL_SOURCE')]:
            s = self.cell(variable + ' =')
            assign = next(n for n in ast.parse(s).body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == variable for t in n.targets))
            self.assertEqual((HERE / name).read_text(encoding='utf-8').strip(), ast.literal_eval(assign.value).strip())
        original = ast.parse(self.cell('def build_reference_config('))
        actual = ast.parse((HERE / 'config.py').read_text(encoding='utf-8'))
        self.assertEqual([ast.dump(n) for n in original.body if isinstance(n, ast.FunctionDef)],
                         [ast.dump(n) for n in actual.body if isinstance(n, ast.FunctionDef)])

    def make_synthetic_result(self, root):
        run_dir, ckpt = root / 'run', root / 'checkpoints'
        step = ckpt / 'global_step_200'
        (run_dir / 'evaluation').mkdir(parents=True)
        (step / 'actor/lora_adapter').mkdir(parents=True)
        for name in ['data.pt', 'actor/lora_adapter/adapter_config.json', 'actor/lora_adapter/adapter_model.safetensors']:
            (step / name).write_bytes(b'fixture')
        (step / 'actor/fsdp_config.json').write_text(json.dumps({'world_size': 4}))
        for rank in range(4):
            for kind in ('model', 'optim', 'extra_state'):
                (step / f'actor/{kind}_world_size_4_rank_{rank}.pt').write_bytes(b'fixture')
        (run_dir / 'data_manifest.json').write_text(json.dumps({'usable_rows': {'evaluation': 2}}))
        (run_dir / 'evaluation/200.jsonl').write_text('\n'.join(json.dumps({'step': 200, 'score': x}) for x in (1, 0)))
        return run_dir, ckpt, step

    def test_complete_four_rank_result_is_collected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, ckpt, _ = self.make_synthetic_result(Path(temporary))
            with contextlib.redirect_stdout(io.StringIO()):
                result = collect_results(ckpt, run_dir, 200, train.MODEL_ID, 'riskpo', 'gsm8k')
            self.assertEqual(result['accuracy'], 0.5)
            self.assertFalse(result['merged_model_export_complete'])

    def test_missing_rank_is_not_a_successful_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, ckpt, step = self.make_synthetic_result(Path(temporary))
            (step / 'actor/model_world_size_4_rank_3.pt').unlink()
            with self.assertRaises(RuntimeError):
                collect_results(ckpt, run_dir, 200, train.MODEL_ID, 'riskpo', 'gsm8k')
            self.assertFalse((run_dir / 'results.json').exists())

    def test_incomplete_evaluation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, ckpt, _ = self.make_synthetic_result(Path(temporary))
            (run_dir / 'evaluation/200.jsonl').write_text(json.dumps({'step': 200, 'score': 1}))
            with self.assertRaises(AssertionError):
                collect_results(ckpt, run_dir, 200, train.MODEL_ID, 'riskpo', 'gsm8k')
            self.assertFalse((run_dir / 'results.json').exists())

    def test_invalid_local_model_does_not_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(RuntimeError):
                train.resolve_model(Path(temporary), offline=True)


if __name__ == '__main__':
    unittest.main()
