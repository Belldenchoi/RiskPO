from pathlib import Path
import re
import datasets

def make_gsm8k_row(example, idx, split):
    question_raw, answer_raw = example['question'], example['answer']
    match = re.search(r'#### (\-?[0-9\.\,]+)', answer_raw)
    assert match is not None, 'GSM8K answer missing #### numeric ground truth'
    solution = match.group(1).replace(',', '')
    instruction = r"Let's think step by step and output the final answer within \boxed{}."
    return {
        'data_source': 'openai/gsm8k',
        'prompt': [{'role': 'user', 'content': question_raw + ' ' + instruction}],
        'ability': 'math',
        'reward_model': {'style': 'rule', 'ground_truth': solution},
        'extra_info': {'split': split, 'index': idx, 'answer': answer_raw, 'question': question_raw},
    }

def prepare_gsm8k(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = [split for split in ('train', 'test') if not (output_dir / (split + '.parquet')).exists()]
    if not missing:
        print('Reuse existing GSM8K train/test; contents checked in step 8.')
        return
    dataset = datasets.load_dataset('openai/gsm8k', 'main')
    for split in missing:
        processed = dataset[split].map(make_gsm8k_row, with_indices=True, fn_kwargs={'split': split})
        processed.to_parquet(str(output_dir / (split + '.parquet')))
        print('GSM8K', split, 'rows:', len(processed))


def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(description="Prepare the original GSM8K train/test splits")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        missing = [s for s in ("train", "test") if not (args.output_dir / (s + ".parquet")).is_file()]
        if missing:
            raise SystemExit("Offline mode needs preprocessed train.parquet and test.parquet in " + str(args.output_dir))
    prepare_gsm8k(args.output_dir)


if __name__ == "__main__":
    main()
