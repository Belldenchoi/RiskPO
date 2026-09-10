"""Export a trained LoRA adapter as a standalone Hugging Face model on CPU.

Only invoked after training exits. No API service or GPU is required for export.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def export_model(base, adapter, output, thinking, step):
    import torch
    from peft import PeftModel
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base, adapter, output = Path(base).resolve(), Path(adapter).resolve(), Path(output).resolve()
    if not (base / 'config.json').is_file():
        raise ValueError('Use the exact local base snapshot used during training')
    for name in ('adapter_config.json', 'adapter_model.safetensors'):
        if not (adapter / name).is_file():
            raise FileNotFoundError(adapter / name)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite exported model: {output}')
    partial = output.with_name(output.name + '.partial')
    if partial.exists():
        raise FileExistsError(f'Previous partial export exists; inspect it first: {partial}')
    config = json.loads((adapter / 'adapter_config.json').read_text())
    if config.get('peft_type') != 'LORA':
        raise ValueError('Expected LoRA adapter')
    with safe_open(str(adapter / 'adapter_model.safetensors'), framework='pt', device='cpu') as tensors:
        keys = list(tensors.keys())
        if not keys or not any('lora_' in k for k in keys):
            raise ValueError('Adapter contains no LoRA tensors')
        for key in keys:
            if not torch.isfinite(tensors.get_tensor(key)).all():
                raise ValueError(f'Non-finite adapter tensor: {key}')
    print('Loading exact training base on CPU for final merge:', base, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base), torch_dtype=torch.bfloat16, device_map={'': 'cpu'},
        low_cpu_mem_usage=True, attn_implementation='eager', local_files_only=True,
    )
    peft_model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    merged = peft_model.merge_and_unload(safe_merge=True)
    if any('lora_' in key for key in merged.state_dict()):
        raise ValueError('Export still contains unmerged LoRA parameters')
    merged.save_pretrained(str(partial), safe_serialization=True, max_shard_size='4GB')
    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
    if thinking is not None:
        template = tokenizer.get_chat_template()
        expected = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': 'What is 1 + 1?'}], tokenize=False,
            add_generation_prompt=True, enable_thinking=thinking,
        )
        tokenizer.chat_template = '{% set enable_thinking = ' + ('true' if thinking else 'false') + ' %}' + template
        actual = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': 'What is 1 + 1?'}], tokenize=False, add_generation_prompt=True)
        if actual != expected:
            raise ValueError('Export tokenizer does not preserve training thinking mode')
    tokenizer.save_pretrained(str(partial))
    if getattr(merged, 'generation_config', None) is not None:
        merged.generation_config.save_pretrained(str(partial))
    del merged, peft_model, model
    gc.collect()
    shards = sorted(partial.glob('*.safetensors'))
    if not shards:
        raise ValueError('No full model weight shards were saved')
    saved_keys = set()
    for shard in shards:
        with safe_open(str(shard), framework='pt', device='cpu') as tensors:
            keys = list(tensors.keys())
            if any('lora_' in key for key in keys):
                raise ValueError('Adapter-only tensor found in merged model')
            if saved_keys.intersection(keys):
                raise ValueError('Duplicate keys across model shards')
            saved_keys.update(keys)
    index = partial / 'model.safetensors.index.json'
    if index.exists():
        weight_map = json.loads(index.read_text())['weight_map']
        if set(weight_map) != saved_keys or set(weight_map.values()) != {p.name for p in shards}:
            raise ValueError('Weight index and actual shards do not match')
    check_tokenizer = AutoTokenizer.from_pretrained(str(partial), local_files_only=True)
    if not check_tokenizer.encode('export validation'):
        raise ValueError('Cannot use exported tokenizer')
    manifest = {
        'status': 'merged_model_export_verified', 'step': step, 'base_snapshot': str(base),
        'adapter_path': str(adapter), 'adapter_sha256': sha256(adapter / 'adapter_model.safetensors'),
        'enable_thinking': thinking, 'dtype': 'bfloat16',
        'validation': 'safe LoRA merge; readable weight shards/index/tokenizer; no post-export GPU inference test',
        'files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)}
                  for p in sorted(partial.iterdir()) if p.is_file()},
    }
    (partial / 'export_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    partial.rename(output)
    print('FULL_MODEL_EXPORTED:', output, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--thinking', choices=['true', 'false', 'default'], default='false')
    parser.add_argument('--step', required=True, type=int)
    args = parser.parse_args()
    thinking = {'true': True, 'false': False, 'default': None}[args.thinking]
    export_model(args.base, args.adapter, args.output, thinking, args.step)


if __name__ == '__main__':
    main()
