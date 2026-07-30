from _common import *

log = logging.getLogger(__name__)

from collections import defaultdict

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

finetuned_models = {
    dataset_name: GPT2ForSequenceClassification.from_pretrained(
        RESULTS_DIR / "gpt2" / dataset_name / "checkpoint-latest"
    )
    for dataset_name in glue_datasets.keys()
}
finetuned_backbones = {
    dataset_name: finetuned_models[dataset_name].transformer
    for dataset_name in finetuned_models.keys()
}


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


if __name__ == "__main__":
    results = defaultdict(list)

    for model_name in (
        pbar := tqdm(list(finetuned_models.keys())[-2:], "Evaluating model")
    ):
        pbar.set_description(f"Evaluating {model_name}")
        results["model"].append(model_name)
        for dataset_name in glue_datasets.keys():
            val_loader = get_val_dataloader(dataset_name)
            model = deepcopy(finetuned_models[model_name])
            acc = eval_model(model, val_loader)
            results[dataset_name].append(acc)

        results_df = pd.DataFrame(results)
        log.info(results_df)

    results_df.to_csv(RESULTS_DIR / "gpt2" / "individuals.csv", index=False)
