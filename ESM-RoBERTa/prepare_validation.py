import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


GLUE_TASK_MAP = {
    "cola": 0,
    "sst2": 1,
    "mrpc": 2,
    "stsb": 3,
    "qqp": 4,
    "mnli": 5,
    "qnli": 6,
    "rte": 7,
}

TASK_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
}


def prepare_glue_validation(
    model_name: str = "roberta-base",
    output_path: str = "data/validation.json",
):
    """
    Tokenize GLUE validation sets and save as a single JSON file.

    Each sample contains: input_ids, attention_mask, label, idx, dataset_ids,
    and the original sentence(s).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    all_data = []

    for task_name, dataset_id in GLUE_TASK_MAP.items():
        split = "validation_matched" if task_name == "mnli" else "validation"
        dataset = load_dataset("glue", task_name, split=split)
        sentence1_key, sentence2_key = TASK_KEYS[task_name]

        for example in tqdm.tqdm(dataset, desc=f"Processing {task_name}"):
            args = (example[sentence1_key],) if sentence2_key is None \
                else (example[sentence1_key], example[sentence2_key])
            tokenized = tokenizer(*args, truncation=True, padding=False)

            entry = {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "label": example["label"],
                "idx": example["idx"],
                "dataset_ids": dataset_id,
            }
            if sentence2_key is None:
                entry["sentence"] = example[sentence1_key]
            else:
                entry["sentence1"] = example[sentence1_key]
                entry["sentence2"] = example[sentence2_key]

            all_data.append(entry)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False)

    # Print per-task statistics
    task_counts = {}
    for item in all_data:
        tid = item["dataset_ids"]
        task_counts[tid] = task_counts.get(tid, 0) + 1
    print(f"\nSaved {len(all_data)} samples to {output_path}")
    for tid, count in sorted(task_counts.items()):
        task_name = [k for k, v in GLUE_TASK_MAP.items() if v == tid][0]
        print(f"  {task_name}: {count}")
    return output_path


if __name__ == "__main__":
    prepare_glue_validation()
