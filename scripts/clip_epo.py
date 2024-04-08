from _common import *

log = logging.getLogger(__name__)

from collections import defaultdict
from typing import cast

import lightning as L
import open_clip.model
from clip_checkpoint_path import (
    CHECKPOINT_DIR,
    finetuned_model_path,
    pretrained_model_path,
)
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.wrappers import _FabricModule
from lightning.pytorch.loggers import TensorBoardLogger
from open_clip.model import CLIP, ResidualAttentionBlock, VisualTransformer
from torch.utils.data import DataLoader

from src.clip_eval import eval_single_dataset
from src.datasets.common import maybe_dictionarize
from src.heads import get_classification_head
from src.modeling import ClassificationHead, ImageEncoder
from src.module.dict_moe import ParetoWeightEnsemblingModule
from src.module.utils import get_by_name, print_trainable_parameters, set_by_name
from src.phn.solvers import EPOSolver
from src.task_vectors import StateDict, TaskVector, state_dict_mean
from src.utils import timeit_context
from torch.utils.data.distributed import DistributedSampler
from lightning.pytorch.callbacks import RichModelSummary, DeviceStatsMonitor


class Program:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        cfg.save = str(CHECKPOINT_DIR / cfg.model)
        cfg.data_location = str(DATA_DIR)
        if cfg.model is None:
            raise ValueError("model must be specified")

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
        cfg.result_dir = str(self.result_dir)
        log.info(f"result_dir: {self.result_dir}")

        logger = TensorBoardLogger(save_dir=self.result_dir, name="tb_logs", version="")
        self.fabric = L.Fabric(
            accelerator="cuda",
            devices=cfg.num_devices,
            loggers=logger,
            strategy="ddp" if cfg.num_devices > 1 else "auto",
            # strategy=self._fsdp_strategy() if cfg.num_devices > 1 else "auto",
            callbacks=[DeviceStatsMonitor()],
        )
        self.fabric.launch()

    def run(self):
        self.load_clip_models()
        self.load_datasets()

        if not os.path.exists(
            self.result_dir / "checkpoints" / f"model_step={self.cfg.num_steps}.ckpt"
        ):
            self.epo_train()
        self.evaluate()

    def epo_train(self):
        cfg = self.cfg
        num_objectives = len(
            self.finetuned_models
        )  # fine-tuned models are not used in this method
        epo_solver = EPOSolver(n_tasks=num_objectives, n_params=None)
        ray = torch.tensor([1 / num_objectives] * num_objectives).to(self.fabric.device)

        model = deepcopy(self.pretrained_model)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr
        )
        model, optimizer = self.fabric.setup(model, optimizer)
        self.classification_heads = {
            t: h.to(self.fabric.device) for t, h in self.classification_heads.items()
        }
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=cfg.num_steps, eta_min=cfg.lr * 0.1
        )

        model.train()
        device = self.fabric.device
        for step_idx in tqdm(
            range(1, 1 + cfg.num_steps), "training", dynamic_ncols=True
        ):
            losses = []
            for dataset_idx, dataset_name in enumerate(cfg.seen_datasets):
                batch = next(self.train_loader_iters[dataset_idx])
                batch = maybe_dictionarize(batch)
                x = batch["images"].to(device)
                y = batch["labels"].to(device)

                features = model(x)
                logits = self.classification_heads[dataset_name](features)

                _loss = F.cross_entropy(logits, y)
                losses.append(_loss)

            losses = torch.stack(losses)
            loss = epo_solver.get_weighted_loss(
                losses,
                ray,
                tuple(filter(lambda p: p.requires_grad, model.parameters())),
            )

            optimizer.zero_grad()
            self.fabric.backward(loss)
            optimizer.step()

            lr_scheduler.step()

            self.fabric.log("loss", loss.item(), step=step_idx)

            if step_idx % cfg.save_interval == 0:
                (self.result_dir / "checkpoints").mkdir(exist_ok=True, parents=True)
                self.fabric.save(
                    self.result_dir / "checkpoints" / f"model_step={step_idx}.ckpt",
                    {"model": model},
                )

    @torch.inference_mode()
    def evaluate(self):
        results = defaultdict(list)

        cfg = self.cfg
        assert cfg.num_devices == 1
        num_objectives = len(self.finetuned_models)
        device = self.fabric.device
        self.classification_heads = {
            t: h.to(self.fabric.device) for t, h in self.classification_heads.items()
        }
        for step_idx in tqdm(
            [8000, 6000, 4000, 2000],
            "evaluating",
            leave=False,
        ):
            log.info(
                f"evaluating step={step_idx}, lodding model from {self.result_dir / 'checkpoints'/ f'model_step={step_idx}.ckpt'}"
            )
            state_dict = torch.load(
                self.result_dir / "checkpoints" / f"model_step={step_idx}.ckpt",
                map_location="cpu",
            )
            if len(state_dict) == 1 and "model" in state_dict:
                state_dict = state_dict["model"]
            model = deepcopy(self.pretrained_model)
            model.load_state_dict(state_dict)
            model = self.fabric.setup_module(model)
            model.eval()

            if True:
                uniform_grid = np.array(
                    [[1 / num_objectives] * num_objectives], dtype=np.float32
                )
            for ray_idx, ray in tqdm(enumerate(uniform_grid), "evaluating samples"):
                results["step"].append(step_idx)
                # sample a preference ray
                for i in range(len(ray)):
                    results[f"ray_{i}"].append(ray[i])
                ray = torch.from_numpy(ray).to(device)
                ParetoWeightEnsemblingModule.set_preferenec_vector(model, ray)

                for dataset_idx, dataset_name in enumerate(
                    tqdm(
                        self.cfg.test_datasets,
                        "evaluating datasets",
                        leave=False,
                    )
                ):
                    test_loader = self.test_loaders[dataset_idx]
                    TOTAL_CORRECT = 0
                    TOTAL_COUNT = 0
                    for batch_idx, batch in (
                        pbar := tqdm(
                            enumerate(test_loader),
                            f"evaluate {dataset_name}",
                            leave=False,
                        )
                    ):
                        batch = maybe_dictionarize(batch)
                        x = batch["images"].to(self.fabric.device)
                        y = batch["labels"].to(self.fabric.device)

                        features = model(x)
                        logits = self.classification_heads[dataset_name](features)
                        preds = logits.argmax(-1)

                        correct = (preds == y).sum().item()
                        TOTAL_CORRECT += correct
                        TOTAL_COUNT += len(y)
                        acc = TOTAL_CORRECT / TOTAL_COUNT
                        pbar.set_postfix_str(f"acc={acc:.2f}")

                        if cfg.quick_evaluation and batch_idx > 20:
                            break
                    results[dataset_name].append(acc)

                (df := pd.DataFrame(results)).to_csv(
                    self.result_dir / "result.csv", index=False
                )
                log.info(df)

    def load_clip_models(self):
        """
        Loads the pretrained CLIP model and the fine-tuned models for each dataset specified in the configuration.
        It first loads the pretrained model from the path specified in the configuration.
        It then loads each fine-tuned model from the path specified in the configuration,
        using the name of the dataset to construct the path.
        Finally, it sets up the classification heads for each dataset, using the configuration and the name of the dataset.

        Side Effects:
            Sets the instance variables `pretrained_model`, `finetuned_models`, and `classification_heads`.
        """
        cfg = self.cfg

        # load pretrained and fine-tuned model
        with timeit_context():
            log.info("load models")
            pretrained_model: ImageEncoder = torch.load(
                pretrained_model_path(cfg.model), map_location="cpu"
            )
            finetuned_models: Dict[ImageEncoder] = {}
            for dataset_name in track(
                (
                    cfg.seen_datasets
                    if cfg.model_seen_datasets is None
                    else cfg.model_seen_datasets
                ),
                "loading finetuned models",
            ):
                log.info(f"loading finetuned model for {dataset_name}")
                finetuned_models[dataset_name] = torch.load(
                    finetuned_model_path(cfg.model, dataset_name),
                    map_location="cpu",
                )

        self.pretrained_model = pretrained_model
        self.finetuned_models = finetuned_models
        self.classification_heads = {
            dataset_name: (
                get_classification_head(cfg, dataset_name).requires_grad_(False).eval()
            )
            for dataset_name in cfg.test_datasets
        }

        log.info("pretrained model statistics:")
        print_trainable_parameters(self.pretrained_model)

    def load_datasets(self):
        """
        Loads the datasets specified in the configuration.

        It first imports the necessary modules and sets up a basic transform for the images.
        It then loads each dataset specified in the configuration, applies the basic transform,
        and sets the location, batch size, and number of workers from the configuration.

        The test dataset from each loaded dataset is added to the list of test datasets.
        It then sets up the data loaders for the test datasets, both with
        and without shuffling, and creates an iterator for each shuffled test loader.

        Side Effects:
            Sets the instance variables `test_datasets`, `test_loaders`, `shuffled_test_loaders`, and
            `shuffled_test_loader_iters`.
        """
        cfg = self.cfg
        cfg.batch_size = cfg.batch_size // cfg.num_devices

        if self.cfg.corruption is None:
            from src.datasets.registry import get_dataset
        else:
            from src.datasets.corruption.registry import get_dataset

        cfg = self.cfg

        dataset_kwargs = dict(
            location=cfg.data_location,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        if self.cfg.corruption is not None:
            dataset_kwargs["corruption"] = self.cfg.corruption
        datasets = [
            get_dataset(
                dataset_name,
                self.pretrained_model.val_preprocess,
                **dataset_kwargs,
            )
            for dataset_name in cfg.test_datasets
        ]
        self.test_datasets = [d.test_dataset for d in datasets]
        self.train_loaders = [
            DataLoader(
                d.train_dataset,
                # Setting shuffle=False in the dataloader constructor prevents the data from being
                # shuffled again after it has already been shuffled by the distributed sampler.
                shuffle=False if cfg.num_devices > 1 else True,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                pin_memory=False,
                sampler=(
                    DistributedSampler(d.train_dataset, shuffle=True)
                    if cfg.num_devices > 1
                    else None
                ),
            )
            for d in datasets
        ]
        self.train_loader_iters = [iter(itertools.cycle(d)) for d in self.train_loaders]
        self.test_loaders = [
            DataLoader(
                d,
                shuffle=False,
                batch_size=cfg.eval_batch_size,
                num_workers=cfg.num_workers,
                pin_memory=False,
            )
            for d in self.test_datasets
        ]
        self.shuffled_test_loaders = [
            DataLoader(
                d,
                shuffle=True,
                batch_size=cfg.tta_batch_size,
                num_workers=cfg.num_workers,
                pin_memory=False,
            )
            for d in self.test_datasets
        ]
        self.shuffled_test_loader_iters = [
            iter(itertools.cycle(d)) for d in self.shuffled_test_loaders
        ]


@hydra.main(
    config_path=str(CONFIG_DIR),
    config_name="clip_pareto_moe",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    program = Program(cfg)
    program.run()


if __name__ == "__main__":
    main()
