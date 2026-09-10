import json

def collect_results(CKPT_DIR, RUN_DIR, TOTAL_STEPS, MODEL_ID, METHOD, DATASET, REQUIRED_GPUS=4):
    FINAL_CHECKPOINT = CKPT_DIR / f"global_step_{TOTAL_STEPS}"
    ADAPTER_DIR = FINAL_CHECKPOINT / "actor/lora_adapter"
    required = ["data.pt", "actor/fsdp_config.json", "actor/lora_adapter/adapter_config.json",
                "actor/lora_adapter/adapter_model.safetensors"]
    required += [f"actor/{kind}_world_size_{REQUIRED_GPUS}_rank_{rank}.pt"
                 for rank in range(REQUIRED_GPUS) for kind in ("model", "optim", "extra_state")]
    if not all((FINAL_CHECKPOINT / p).is_file() and (FINAL_CHECKPOINT / p).stat().st_size for p in required):
        raise RuntimeError("Full run did not produce the complete final checkpoint; do not export as completed")
    fsdp = json.loads((FINAL_CHECKPOINT / "actor/fsdp_config.json").read_text())
    assert fsdp["world_size"] == REQUIRED_GPUS
    EVAL_FILE = RUN_DIR / "evaluation" / f"{TOTAL_STEPS}.jsonl"
    if not EVAL_FILE.is_file():
        raise RuntimeError("Final evaluation output is missing")
    evaluation = [json.loads(line) for line in EVAL_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = json.loads((RUN_DIR / "data_manifest.json").read_text())
    assert len(evaluation) == manifest["usable_rows"]["evaluation"], "Evaluation is incomplete"
    assert all(row["step"] == TOTAL_STEPS for row in evaluation)
    scores = [float(row["score"]) for row in evaluation]
    import math
    assert scores and all(math.isfinite(s) for s in scores)
    results = {
        "training_complete": True, "step": TOTAL_STEPS, "model": MODEL_ID, "method": METHOD,
        "dataset": DATASET, "evaluation_samples": len(scores), "mean_score": sum(scores) / len(scores),
        "checkpoint": str(FINAL_CHECKPOINT), "adapter": str(ADAPTER_DIR),
        "scorer": "final_numeric_v2 for GSM8K; repo default for other datasets",
        "evaluation_file": str(EVAL_FILE), "merged_model_export_complete": False,
    }
    if DATASET == "gsm8k":
        assert all(s in (0.0, 1.0) for s in scores)
        results.update(correct=sum(s == 1 for s in scores), accuracy=sum(scores) / len(scores))
    (RUN_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print("FULL TRAINING AND FINAL EVALUATION COMPLETE. Next cell exports standalone model.")
    return results
