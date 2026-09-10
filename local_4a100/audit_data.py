import argparse
import json
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("audit_file", type=Path)
args = parser.parse_args()
AUDIT = json.loads(args.audit_file.read_text(encoding="utf-8"))

from pathlib import Path
import hashlib
import json
import importlib.util
import pandas as pd
from hydra import initialize_config_dir, compose
from transformers import AutoTokenizer
from verl.trainer.main_ppo import create_rl_dataset

with initialize_config_dir(config_dir=AUDIT["config_dir"], version_base=None):
    cfg = compose(config_name=AUDIT["config_name"])
spec = importlib.util.spec_from_file_location("riskpo_audit_reward", cfg.custom_reward_function.path)
reward_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reward_module)
score_fn = getattr(reward_module, cfg.custom_reward_function.name)
assert reward_module.GSM8K_SCORER_VERSION == "final_numeric_v2"
tokenizer = AutoTokenizer.from_pretrained(cfg.actor_rollout_ref.model.path)
manifest = {"files": [], "usable_rows": {}}
for role, paths in [("train", cfg.data.train_files), ("evaluation", cfg.data.val_files)]:
    for filename in paths:
        path = Path(filename)
        frame = pd.read_parquet(path)
        if AUDIT["dataset"] == "gsm8k":
            expected_split = 'train' if role == 'train' else 'test'
            assert len(paths) == 1 and path.name == expected_split + '.parquet'
            assert set(frame['data_source']) == {'openai/gsm8k'}, 'Unexpected dataset in GSM8K-only run'
            assert all(info['split'] == expected_split for info in frame['extra_info']), 'Wrong GSM8K split'
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        item = {"role": role, "path": str(path), "rows": len(frame),
                "sha256": digest.hexdigest(),
                "sources": {str(k): int(v) for k, v in frame["data_source"].value_counts().items()}}
        manifest["files"].append(item)
        print(item)
        gsm = frame[frame["data_source"] == "openai/gsm8k"]
        if len(gsm):
            truth = str(gsm.iloc[0]["reward_model"]["ground_truth"])
            for response in [r"\boxed{" + truth + "}", "#### " + truth]:
                assert score_fn("openai/gsm8k", response, truth) == 1, (response, truth)
            print("GSM8K boxed/#### checks passed:", reward_module.GSM8K_SCORER_VERSION)
    dataset = create_rl_dataset(list(paths), cfg.data, tokenizer, None, is_train=(role == "train"))
    manifest["usable_rows"][role] = len(dataset)
    assert len(dataset) > 0, f"{role} empty after filtering"
    if cfg.data.get("enable_thinking") is not None:
        enabled = cfg.data.enable_thinking
        row = dataset[0]
        messages = dataset.dataframe[0][cfg.data.prompt_key]
        expected_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enabled)
        expected_ids = tokenizer.encode(expected_text, add_special_tokens=False)
        assert list(row["raw_prompt_ids"]) == expected_ids, 'Rollout prompt does not match thinking mode'
        manifest.setdefault("thinking_checks", {})[role] = {
            "enable_thinking": enabled, "raw_prompt_ids_match": True}
        print(role, "raw_prompt_ids check passed; enable_thinking =", enabled)
assert manifest["usable_rows"]["train"] >= cfg.data.train_batch_size
(Path(AUDIT["run_dir"]) / "data_manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)
print("After prompt filtering:", manifest["usable_rows"])
