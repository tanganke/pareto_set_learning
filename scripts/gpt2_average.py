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
from scripts.gpt2_evaluate import get_val_dataloader, eval_model
from abc import ABC, abstractmethod
import argparse
from fusionlib.merge.average import simple_average


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["mrpc", "mnli", "cola", "sst2", "qnli", "qqp", "rte"],
    )
    parser.add_argument("--version", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    print(args)

    log.info("Loading pretrained backbone")
    pretrained_backbone = GPT2ForSequenceClassification.from_pretrained(
        CACHE_DIR / "models" / "gpt2"
    ).transformer
    finetuned_models = {}
    for task in args.tasks:
        log.info(f"Loading finetuned model for {task}")
        finetuned_models[task] = GPT2ForSequenceClassification.from_pretrained(
            str(RESULTS_DIR / args.model / task / "checkpoint-latest")
        )
    finetuned_backbones = {
        dataset_name: finetuned_models[dataset_name].transformer
        for dataset_name in finetuned_models.keys()
    }

    result_dir = RESULTS_DIR / args.model / "average"
    if args.version is not None:
        result_dir = result_dir / f"version_{args.version}"

    results = defaultdict(list)
    with torch.no_grad():
        merged_backbone = simple_average(
            [finetuned_backbones[task] for task in args.tasks]
        )

        for task in args.tasks:
            model = deepcopy(finetuned_models[task])
            model.transformer = merged_backbone
            model = model.to("cuda")

            val_dataloader = get_val_dataloader(task)
            acc = eval_model(model, val_dataloader)
            results[task].append(acc)

        df = pd.DataFrame(results)
        print(df)
        result_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(result_dir / "results.csv")


if __name__ == "__main__":
    main()
