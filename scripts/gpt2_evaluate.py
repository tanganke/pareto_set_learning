from _common import *

log = logging.getLogger(__name__)

from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from datasets import load_dataset, load_from_disk
from scripts.gpt2_finetune import glue_datasets, pretrained_model, tokenizer


def get_val_dataset(dataset_name: str):
    dataset = glue_datasets[dataset_name]
    if "validation" in dataset:
        return dataset["validation"]
    elif "validation_matched" in dataset:
        return dataset["validation_matched"]
    else:
        raise RuntimeError()


def get_val_dataloader(dataset_name: str, shuffle=False, batch_size=16):
    val_dataset = get_val_dataset(dataset_name)
    val_loader = DataLoader(
        val_dataset,
        collate_fn=default_data_collator,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return val_loader


def to_device(obj, device=None, fabric=None):
    if fabric is not None:
        return fabric.to_device(obj)
    else:
        return obj.to(device)


@torch.no_grad()
def eval_model(
    model: GPT2ForSequenceClassification, dataloader: DataLoader, fabric=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = to_device(model, device, fabric)
    model.eval()
    total_correct = 0
    total_count = 0
    for batch in (pbar := tqdm(dataloader, leave=False)):
        input_ids = to_device(batch["input_ids"], device, fabric)
        attention_mask = to_device(batch["attention_mask"], device, fabric)
        labels = to_device(batch["labels"], device, fabric)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()
        total_correct += correct
        total_count += len(labels)
        pbar.set_postfix_str(f"acc={total_correct / total_count:.2f}")
    return total_correct / total_count


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["mrpc", "mnli", "cola", "sst2", "qnli", "qqp", "rte"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = GPT2ForSequenceClassification.from_pretrained(args.model_path)
    results = defaultdict(list)

    for dataset_name in tqdm(args.dataset, desc="Evaluating"):
        val_loader = get_val_dataloader(dataset_name)
        acc = eval_model(model, val_loader)
        results[dataset_name].append(acc)
        log.info(f"dataset: {dataset_name}, Accuracy: {acc:.2f}")

    # save results to args.model_path / "results.txt"
    df = pd.DataFrame(results)
    print(df)
    df.to_csv(os.path.join(args.model_path, f"results.csv"), index=False)
