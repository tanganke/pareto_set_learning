from _common import *

log = logging.getLogger(__name__)

import torch
from transformers import (
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)

from datasets import load_dataset, load_from_disk

# Load the pre-trained GPT-2 model and tokenizer
log.info("Loading GPT-2 model and tokenizer")
model_name = str(CACHE_DIR / "models/gpt2")
pretrained_model = GPT2ForSequenceClassification.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.model_max_length = 512
if tokenizer.pad_token is None:
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    elif tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError
if pretrained_model.config.pad_token_id is None:
    pretrained_model.config.pad_token_id = tokenizer.pad_token_id


def cache_dataset(func, model_name="gpt2"):
    def wrapper(*args, **kwargs):
        cache_path = RESULTS_DIR / model_name / f"_{func.__name__}_cached"
        if cache_path.parent.exists() is False:
            cache_path.parent.mkdir(parents=True)
        if cache_path.exists():
            dataset = load_from_disk(str(cache_path))
        else:
            dataset = func(*args, **kwargs)
            dataset.save_to_disk(cache_path)
        return dataset

    return wrapper


# Tokenize and convert examples to features
def mrpc_tokenize_function(examples):
    inputs = tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


@cache_dataset
def load_mrpc_dataset():
    dataset = load_dataset("glue", "mrpc")
    dataset = dataset.map(
        mrpc_tokenize_function, batched=True, remove_columns=["sentence1", "sentence2"]
    )
    return dataset


@cache_dataset
def load_rte_dataset():
    dataset = load_dataset("glue", "rte")
    dataset = dataset.map(
        mrpc_tokenize_function, batched=True, remove_columns=["sentence1", "sentence2"]
    )
    return dataset


@cache_dataset
def load_wnli_dataset():
    dataset = load_dataset("glue", "wnli")
    dataset = dataset.map(
        mrpc_tokenize_function, batched=True, remove_columns=["sentence1", "sentence2"]
    )
    return dataset


def mnli_tokenize_function(examples):
    inputs = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


@cache_dataset
def load_mnli_dataset():
    dataset = load_dataset("glue", "mnli")
    dataset = dataset.map(
        mnli_tokenize_function, batched=True, remove_columns=["premise", "hypothesis"]
    )
    return dataset


def cola_tokenize_function(examples):
    inputs = tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


@cache_dataset
def load_cola_dataset():
    dataset = load_dataset("glue", "cola")
    dataset = dataset.map(
        cola_tokenize_function, batched=True, remove_columns=["sentence"]
    )
    return dataset


@cache_dataset
def load_sst2_dataset():
    dataset = load_dataset("glue", "sst2")
    dataset = dataset.map(
        cola_tokenize_function, batched=True, remove_columns=["sentence"]
    )
    return dataset


def qnli_tokenize_function(examples):
    inputs = tokenizer(
        examples["question"],
        examples["sentence"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


@cache_dataset
def load_qnli_dataset():
    dataset = load_dataset("glue", "qnli")
    dataset = dataset.map(
        qnli_tokenize_function, batched=True, remove_columns=["question", "sentence"]
    )
    return dataset


def qqp_tokenize_function(examples):
    inputs = tokenizer(
        examples["question1"],
        examples["question2"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


@cache_dataset
def load_qqp_dataset():
    dataset = load_dataset("glue", "qqp")
    dataset = dataset.map(
        qqp_tokenize_function, batched=True, remove_columns=["question1", "question2"]
    )
    return dataset


glue_dataset_loaders = {
    "mrpc": load_mrpc_dataset,
    "mnli": load_mnli_dataset,
    "cola": load_cola_dataset,
    "sst2": load_sst2_dataset,
    "qnli": load_qnli_dataset,
    "qqp": load_qqp_dataset,
    "rte": load_rte_dataset,
    # "wnli": load_wnli_dataset,
}

# Load the GLUE dataset
log.info("Loading GLUE dataset")
glue_datasets = {name: glue_dataset_loaders[name]() for name in glue_dataset_loaders}


# Define training arguments
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mrpc")
    parser.add_argument("--num_epochs", type=int, default=3)
    args = parser.parse_args()

    dataset_name = args.dataset

    log.info(f"Training on {dataset_name}")
    if dataset_name == "mnli":
        model_config = pretrained_model.config
        model_config.num_labels = 3
        model = GPT2ForSequenceClassification(model_config)
        model.transformer = deepcopy(pretrained_model.transformer)
    else:
        model = deepcopy(pretrained_model)

    training_args = TrainingArguments(
        output_dir=str(RESULTS_DIR / "gpt2" / dataset_name),
        save_strategy="epoch",
        eval_steps=500,
        num_train_epochs=args.num_epochs,
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=1e-9,
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=glue_datasets[dataset_name]["train"],
    )

    # Fine-tune the model
    trainer.train()

    model.save_pretrained(RESULTS_DIR / "gpt2" / dataset_name / "checkpoint-latest")
    tokenizer.save_pretrained(RESULTS_DIR / "gpt2" / dataset_name / "checkpoint-latest")
