from _common import *

log = logging.getLogger(__name__)

from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
import pandas as pd
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from transformers import (
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
    GPT2Model,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from src.module.dict_moe import ParetoWeightEnsemblingModule

from datasets import load_dataset, load_from_disk
from scripts.gpt2_evaluate import get_val_dataloader, eval_model
from scripts.gpt2_finetune import glue_datasets, pretrained_model, tokenizer
from lightning.pytorch.callbacks import (
    RichModelSummary,
    DeviceStatsMonitor,
    LearningRateMonitor,
)
from fusionlib.merge.task_arithmetic import task_arithmetic_merge_modules
from src.utils import timeit_context
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from src.module.utils import print_trainable_parameters
from torch.utils.data.distributed import DistributedSampler
from lightning.fabric.strategies import FSDPStrategy, DDPStrategy
from lightning.fabric.strategies.dp import DataParallelStrategy


def generate_simplex_grid(n, m):
    """
    Generate a uniform grid of points on the n-dimensional simplex.

    Args:
        n (int): The dimension of the simplex.
        m (int): The number of grid points along each dimension.

    Returns:
        list: A list of n-dimensional vectors representing the grid points.
    """
    m = m - 1
    # **Generate all combinations of indices summing up to m**
    indices = list(itertools.combinations_with_replacement(range(m + 1), n - 1))

    # **Initialize an empty list to store the grid points**
    grid_points = []

    # **Iterate over each combination of indices**
    for idx in indices:
        # **Append 0 and m to the indices**
        extended_idx = [0] + list(idx) + [m]

        # **Compute the vector components by taking the differences between consecutive indices and dividing by m**
        point = [(extended_idx[i + 1] - extended_idx[i]) / m for i in range(n)]
        grid_points.append(point)

    return np.array(grid_points, dtype=np.float32)


class GPT2ParetoMoEProgram(ABC):
    def __init__(self, cfg: DictConfig, __file__: str):
        self.cfg = cfg
        if cfg.model is None:
            raise ValueError("model must be specified")

        # setup the result directory
        if cfg.version is None:
            for version_idx in itertools.count():
                self.result_dir = (
                    RESULTS_DIR
                    / os.path.basename(__file__).split(".")[0]
                    / f"version_{version_idx}"
                )
                if not self.result_dir.exists():
                    break
        else:
            self.result_dir = (
                RESULTS_DIR
                / os.path.basename(__file__).split(".")[0]
                / f"version_{cfg.version}"
            )

        # setup fabric
        logger = TensorBoardLogger(save_dir=self.result_dir, name="tb_logs", version="")
        self.fabric = L.Fabric(
            accelerator="cuda",
            devices=cfg.num_devices,
            loggers=logger,
            strategy=(DDPStrategy() if cfg.num_devices > 1 else "auto"),
            # strategy=self._fsdp_strategy() if cfg.num_devices > 1 else "auto",
            callbacks=[DeviceStatsMonitor(), LearningRateMonitor("step")],
        )
        self.fabric.launch()

    def run(self):
        cfg = self.cfg

        self.load_model()
        self.load_datasets()

        if cfg.train:
            self.train()
        if cfg.evaluate:
            self.evaluate()

    def load_model(self):
        cfg = self.cfg

        log.info("Loading pretrained model")
        self.pretrained_model = pretrained_model
        pretrained_backbone = self.pretrained_model.transformer

        self.finetuned_models = {}
        for task in cfg.tasks:
            log.info(f"Loading finetuned model for task {task}")
            self.finetuned_models[task] = GPT2ForSequenceClassification.from_pretrained(
                str(RESULTS_DIR / "gpt2" / task.lower() / "checkpoint-latest")
            )

        self.finetuned_backbone = {
            task: cast(GPT2Model, model.transformer)
            for task, model in self.finetuned_models.items()
        }
        for task in cfg.tasks:
            self.finetuned_models[task].transformer = None

        with timeit_context("building model"):
            if self.cfg.partial:
                # weight ensembling only the MLPs, merge the remaining layers using task arithmetic

                # model merging
                model: GPT2Model = task_arithmetic_merge_modules(
                    pretrained_backbone,
                    self.finetuned_backbone.values(),
                    scaling_coef=cfg.init_lambda,
                )

                # fix the model weights
                model.requires_grad_(False)

                for layer_idx in range(pretrained_model.config.n_layer):
                    cast(GPT2Block, model.h[layer_idx]).mlp = (
                        ParetoWeightEnsemblingModule(
                            base_model=pretrained_backbone.h[layer_idx].mlp,
                            expert_models=[
                                self.finetuned_backbone[task].h[layer_idx].mlp
                                for task in self.finetuned_backbone.keys()
                            ],
                            init_lambda=cfg.init_lambda,
                            fix_base_model_and_experts=True,
                            router_hidden_layers=cfg.router_hidden_layers,
                        )
                    )
            else:
                raise NotImplementedError(
                    "Full model weight ensembling not implemented"
                )

        self.model = model
        print_trainable_parameters(self.model)

    def load_datasets(self):
        cfg = self.cfg
        assert (
            cfg.batch_size % cfg.num_devices == 0
        ), "Batch size must be divisible by num_devices"
        cfg.batch_size = cfg.batch_size // cfg.num_devices

        log.info("Loading datasets")

        if cfg.train:
            train_datasets = {
                task: glue_datasets[task.lower()]["train"] for task in cfg.tasks
            }

            train_loaders = {
                task: DataLoader(
                    train_datasets[task],
                    collate_fn=default_data_collator,
                    batch_size=cfg.batch_size,
                    shuffle=False if cfg.num_devices > 1 else True,
                    sampler=(
                        DistributedSampler(train_datasets[task], shuffle=True)
                        if cfg.num_devices > 1
                        else None
                    ),
                )
                for task in cfg.tasks
            }

            self.train_datasets = train_datasets
            self.train_loaders = train_loaders
            self.train_loader_iters = [
                iter(itertools.cycle(d)) for d in self.train_loaders.values()
            ]

        if cfg.evaluate:
            self.val_loaders = {
                task: get_val_dataloader(
                    task.lower(),
                    batch_size=cfg.batch_size,
                )
                for task in cfg.tasks
            }

    @torch.no_grad()
    def evaluate(self):
        results = defaultdict(list)
        cfg = self.cfg
        assert cfg.num_devices == 1
        num_objectives = len(self.finetuned_models)
        device = self.fabric.device
        classifiers = {
            task: cast(GPT2ForSequenceClassification, m)
            .score.requires_grad_(False)
            .to(device)
            for task, m in self.finetuned_models.items()
        }
        for step_idx in reversed(
            range(cfg.save_interval, cfg.num_steps + 1, cfg.save_interval)
        ):
            log.info(f"Evaluating step {step_idx}")
            state_dict = torch.load(
                self.result_dir / "checkpoints" / f"model_step={step_idx}.ckpt",
                map_location="cpu",
            )
            if len(state_dict) == 1 and "model" in state_dict:
                state_dict = state_dict["model"]
            model = deepcopy(self.model)
            model.load_state_dict(state_dict)
            model = self.fabric.setup_module(model)
            model.eval()

            forward_model = deepcopy(self.pretrained_model)
            forward_model.transformer = model
            forward_model.score = None

            if cfg.num_evaluation_samples == "equal_weight":
                uniform_grid = np.array(
                    [[1 / num_objectives] * num_objectives], dtype=np.float32
                )
            else:
                uniform_grid = generate_simplex_grid(
                    num_objectives, cfg.num_evaluation_samples
                )

            for ray_idx, ray in tqdm(enumerate(uniform_grid), "evaluating samples"):
                results["step"].append(step_idx)
                # sample a preference ray
                for i in range(len(ray)):
                    results[f"ray_{i}"].append(ray[i])
                ray = torch.from_numpy(ray).to(device)
                ParetoWeightEnsemblingModule.set_preferenec_vector(model, ray)

                for dataset_idx, dataset_name in enumerate(
                    tqdm(cfg.tasks, "evaluating")
                ):
                    forward_model.num_labels = self.finetuned_models[
                        dataset_name
                    ].num_labels
                    forward_model.score = classifiers[dataset_name]
                    acc = eval_model(
                        forward_model,
                        self.val_loaders[dataset_name],
                        self.fabric,
                    )
                    results[dataset_name].append(acc)

                (df := pd.DataFrame(results)).to_csv(
                    self.result_dir / "result.csv", index=False
                )
                log.info(df)

    def compute_loss(self, model: nn.Module, ray: Tensor, losses: Tuple[Tensor]):
        raise NotImplementedError()

    def train(self):
        cfg = self.cfg
        device = self.fabric.device
        log.info("Training")

        # save the configuration
        self.result_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, self.result_dir / "train_config.yaml")

        # setup the model
        num_objectives = len(self.finetuned_backbone)
        backbone = deepcopy(self.model)
        classifiers = {
            task: cast(GPT2ForSequenceClassification, m)
            .score.requires_grad_(False)
            .to(device)
            for task, m in self.finetuned_models.items()
        }
        log.info("Classifiers:")
        for task, classifier in classifiers.items():
            log.info(f"{task}: {classifier}")

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, backbone.parameters()),
            lr=cfg.lr,
        )
        backbone, optimizer = self.fabric.setup(backbone, optimizer)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=cfg.num_steps, eta_min=cfg.lr * 0.1
        )

        # setup forward models, share a common backbone
        forward_model = deepcopy(self.pretrained_model)
        forward_model.requires_grad_(False)
        forward_model.transformer = backbone

        backbone.train()
        for step_idx in tqdm(
            range(1, 1 + cfg.num_steps), "training", dynamic_ncols=True
        ):
            # sample a preference ray
            ray = torch.from_numpy(
                np.random.dirichlet((cfg.alpha,) * num_objectives, 1)
                .astype(np.float32)
                .flatten()
            ).to(device)
            ParetoWeightEnsemblingModule.set_preferenec_vector(backbone, ray)

            losses = []
            for dataset_idx, dataset_name in enumerate(cfg.tasks):
                batch = next(self.train_loader_iters[dataset_idx])
                forward_model.num_labels = self.finetuned_models[
                    dataset_name
                ].num_labels
                outputs = torch.func.functional_call(
                    forward_model,
                    parameter_and_buffer_dicts={
                        "score." + k: v
                        for k, v in classifiers[dataset_name]
                        .state_dict(keep_vars=True)
                        .items()
                    },
                    args=tuple(),
                    kwargs=dict(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device),
                    ),
                    strict=False,
                )
                _loss = outputs.loss
                losses.append(_loss)

            loss = self.compute_loss(backbone, ray, losses)

            optimizer.zero_grad()
            self.fabric.backward(loss)
            optimizer.step()

            lr_scheduler.step()

            self.fabric.log("loss", loss.item(), step=step_idx)

            if step_idx % cfg.save_interval == 0:
                (self.result_dir / "checkpoints").mkdir(exist_ok=True, parents=True)
                self.fabric.save(
                    self.result_dir / "checkpoints" / f"model_step={step_idx}.ckpt",
                    {"model": backbone},
                )
