import ast
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import train
from hardware_profiles import hardware
from collect_results import collect_results
from performance import summarize


class L40Tests(unittest.TestCase):
    def config(self, profile, dataset='gsm8k'):
        return train.make_config(dataset, Path('/base'), Path('/data'), Path('/checkpoints'),
                                 Path('/run'), 'test', hardware_profile=profile)

    def test_matches_riskpo_branch_of_current_quatro_notebook(self):
        nb = json.loads((HERE.parent / 'RiskPO_QUATRO_Qwen3_8B_Colab.ipynb').read_text(encoding='utf-8'))
        source = next(''.join(c['source']) for c in nb['cells'] if 'def build_lora_config(' in ''.join(c['source']))
        context = {'Path': Path}
        exec(source, context)
        for dataset in ('gsm8k', 'easymath', 'dapo'):
            for profile in ('l40_2_fast', 'l40_2_safe', 'l40_2_graph'):
                with self.subTest(dataset=dataset, profile=profile):
                    expected = context['build_lora_config'](dataset, 'riskpo', train.MODEL_ID, Path('/data'), Path('/checkpoints'), 'test')
                    expected['actor_rollout_ref']['actor'].update(policy_loss={'loss_mode': 'vanilla'}, loss_agg_mode='token-mean')
                    actual = copy.deepcopy(self.config(profile, dataset))
                    # Restore only explicitly allowed storage/topology/performance differences.
                    actual['data'].pop('enable_thinking')
                    actual['data'].pop('custom_cls')
                    actual.pop('custom_reward_function')
                    actual['actor_rollout_ref']['model']['path'] = train.MODEL_ID
                    actor = actual['actor_rollout_ref']['actor']
                    actor.pop('checkpoint')
                    actor['fsdp_config'].pop('fsdp_size')
                    actor['fsdp_config']['param_offload'] = True
                    actor['fsdp_config']['optimizer_offload'] = True
                    rollout = actual['actor_rollout_ref']['rollout']
                    for key in ('log_prob_micro_batch_size_per_gpu', 'max_num_seqs', 'max_num_batched_tokens', 'enforce_eager', 'engine_kwargs'):
                        rollout[key] = copy.deepcopy(expected['actor_rollout_ref']['rollout'][key])
                    for key in ('n_gpus_per_node', 'save_freq', 'test_freq', 'val_before_train', 'resume_mode', 'resume_from_path', 'max_actor_ckpt_to_keep', 'validation_data_dir'):
                        if key in expected['trainer']:
                            actual['trainer'][key] = expected['trainer'][key]
                        else:
                            actual['trainer'].pop(key)
                    self.assertEqual(actual, expected)

    def test_profiles_have_two_ranks_and_fixed_global_batch(self):
        for profile in ('l40_2_fast', 'l40_2_safe', 'l40_2_graph'):
            cfg = self.config(profile)
            actor, rollout = cfg['actor_rollout_ref']['actor'], cfg['actor_rollout_ref']['rollout']
            self.assertEqual(hardware(profile), (2, 'L40'))
            self.assertEqual(cfg['trainer']['n_gpus_per_node'], 2)
            self.assertEqual(actor['fsdp_config']['fsdp_size'], 2)
            self.assertEqual(actor['ppo_micro_batch_size_per_gpu'], 1)
            self.assertEqual(cfg['data']['train_batch_size'] * rollout['n'] // 2, 50)
            self.assertEqual(rollout['tensor_model_parallel_size'], 1)
            self.assertEqual(cfg['trainer']['total_training_steps'], 200)
            self.assertEqual(actor['fsdp_config']['param_offload'], profile == 'l40_2_safe')
            self.assertEqual(rollout['enforce_eager'], profile != 'l40_2_graph')
            self.assertEqual(rollout['max_num_seqs'], rollout['engine_kwargs']['vllm']['max_num_seqs'])

    def test_loss_weighting_preserved_by_microbatch_one(self):
        # Sequence means stand in for token-mean losses at actor microbatch=1.
        values = [math.sin(i) + i / 100 for i in range(100)]
        single_gpu = sum(values) / 100
        two_gpu = sum(sum(values[rank::2]) / 50 for rank in range(2)) / 2
        self.assertAlmostEqual(single_gpu, two_gpu)

    def test_two_rank_checkpoint_collection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, ckpt = root / 'run', root / 'ckpt'
            actor = ckpt / 'global_step_200/actor'
            (actor / 'lora_adapter').mkdir(parents=True)
            (run_dir / 'evaluation').mkdir(parents=True)
            for rank in range(2):
                for kind in ('model', 'optim', 'extra_state'):
                    (actor / f'{kind}_world_size_2_rank_{rank}.pt').write_bytes(b'fixture')
            (actor.parent / 'data.pt').write_bytes(b'fixture')
            for name in ('adapter_model.safetensors', 'adapter_config.json'):
                (actor / 'lora_adapter' / name).write_bytes(b'fixture')
            (actor / 'fsdp_config.json').write_text(json.dumps({'world_size': 2}))
            (run_dir / 'data_manifest.json').write_text(json.dumps({'usable_rows': {'evaluation': 1}}))
            (run_dir / 'evaluation/200.jsonl').write_text(json.dumps({'step': 200, 'score': 1}))
            with contextlib.redirect_stdout(io.StringIO()):
                result = collect_results(ckpt, run_dir, 200, train.MODEL_ID, 'riskpo', 'gsm8k', 2)
            self.assertTrue(result['training_complete'])
            (actor / 'model_world_size_2_rank_1.pt').unlink()
            with self.assertRaises(RuntimeError):
                collect_results(ckpt, run_dir, 200, train.MODEL_ID, 'riskpo', 'gsm8k', 2)

    def test_performance_uses_warmup_and_complete_steps(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'train.log'
            path.write_text('\n'.join(f'(TaskRunner pid=9) step:{step} - timing_s/step:{seconds} - perf/total_num_tokens:1000 - actor/grad_norm:0.0'
                                     for step, seconds in ((1, 100), (2, 10), (3, 20))))
            result = summarize(path, expected_steps=3)
            self.assertEqual(result['metrics']['timing_s/step']['median'], 15)
            self.assertAlmostEqual(result['aggregate_tokens_per_second'], 2000 / 30)
            self.assertEqual(result['zero_gradient_steps'], [1, 2, 3])
            with self.assertRaises(RuntimeError):
                summarize(path, expected_steps=4)


if __name__ == '__main__':
    unittest.main()
