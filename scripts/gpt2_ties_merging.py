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
from fusionlib.merge.task_arithmetic import task_arithmetic_merge_modules
from src.ties_merging_utils import *


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

    ptm_check: StateDict = pretrained_backbone.state_dict()
    ft_checks: List[StateDict] = [
        finetuned_backbones[task].state_dict() for task in args.tasks
    ]
    check_parameterNamesMatch(ft_checks + [ptm_check])

    remove_keys = []
    print(f"Flattening out Checkpoints")
    flat_ft = torch.vstack(
        [state_dict_to_vector(check, remove_keys) for check in ft_checks]
    )
    flat_ptm = state_dict_to_vector(ptm_check, remove_keys)

    tv_flat_checks = flat_ft - flat_ptm
    assert check_state_dicts_equal(
        vector_to_state_dict(flat_ptm, ptm_check, remove_keys), ptm_check
    )
    assert all(
        [
            check_state_dicts_equal(
                vector_to_state_dict(flat_ft[i], ptm_check, remove_keys), ft_checks[i]
            )
            for i in range(len(ft_checks))
        ]
    )

    result_dir = RESULTS_DIR / args.model / "ties_merging"
    if args.version is not None:
        result_dir = result_dir / f"version_{args.version}"

    K = 20
    merge_func = "dis-sum"

    results = defaultdict(list)
    for scaling_coef_ in tqdm(np.linspace(0, 1, 11), desc="scaling_coef"):
        results["scaling_coef"].append(scaling_coef_)

        merged_tv = ties_merging(
            tv_flat_checks,
            reset_thresh=K,
            merge_func=merge_func,
        )
        merged_check = flat_ptm + scaling_coef_ * merged_tv
        merged_state_dict = vector_to_state_dict(
            merged_check, ptm_check, remove_keys=remove_keys
        )
        model = deepcopy(finetuned_models[args.tasks[0]])
        model.transformer.load_state_dict(merged_state_dict)
        model = model.to("cuda")

        for task in args.tasks:
            val_dataloader = get_val_dataloader(task)
            acc = eval_model(model, val_dataloader)
            results[task].append(acc)

        df = pd.DataFrame(results)
        print(df)
        result_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(result_dir / "results.csv")


if __name__ == "__main__":
    main()
