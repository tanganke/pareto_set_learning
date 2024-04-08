from _common import *

log = logging.getLogger(__name__)

from collections import defaultdict

from clip_checkpoint_path import (
    CHECKPOINT_DIR,
    finetuned_model_path,
    pretrained_model_path,
)
from fusionlib.merge.average import simple_average
from fusionlib.utils.torch.parameters import check_parameters_all_equal
from torch import Tensor, nn
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from src.clip_eval import eval_single_dataset, eval_single_dataset_preprocess_head
from src.heads import get_classification_head
from src.modeling import ClassificationHead, ImageEncoder
from src.utils import first, timeit_context
from src.datasets.common import maybe_dictionarize


def entropy_loss(logits: Tensor) -> Tensor:
    """
    compute the entropy loss
    """
    probs = nn.functional.softmax(logits, dim=-1)
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    return -torch.mean(torch.sum(probs * log_probs, dim=-1))


class Program:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        cfg.save = str(CHECKPOINT_DIR / cfg.model)
        cfg.data_location = str(DATA_DIR)

        for version_idx in itertools.count():
            self.result_dir = (
                RESULTS_DIR / "clip_entropy_tta" / f"version_{version_idx}"
            )
            if not self.result_dir.exists():
                break
        cfg.result_dir = str(self.result_dir)
        log.info(f"result_dir: {self.result_dir}")

        wandb_logger = WandbLogger(config=OmegaConf.to_container(cfg))
        self.fabric = L.Fabric(accelerator="cuda", devices=1, loggers=wandb_logger)
        self.fabric.launch()

    def run(self):
        self.load_model()
        self.load_datasets()

        self.entropy_tta()

        self.cleanup()

    def entropy_tta(self):
        cfg = self.cfg
        model = self.model
        classification_heads = self.classification_heads
        infty_test_loaders = self.infty_test_loaders
        infty_test_loaders_iters = {
            task_name: iter(loader) for task_name, loader in infty_test_loaders.items()
        }

        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        optimizer: torch.optim.Adam = self.fabric.setup_optimizers(optimizer)

        for step_idx in (
            pabr := tqdm(range(1, 1 + cfg.num_steps), "entropy tta", dynamic_ncols=True)
        ):
            # sum the entropy loss over all tasks
            total_loss = 0
            total_correct = 0
            total_count = 0
            for task_name in cfg.test_tasks:
                # get the next batch of data
                batch = next(infty_test_loaders_iters[task_name])
                batch = maybe_dictionarize(batch)

                x = self.fabric.to_device(batch["images"])
                # forward pass
                features = model(x)
                logits = classification_heads[task_name](features)

                # compute the entropy loss
                loss = entropy_loss(logits)
                total_loss += loss

                # --- for logging ---
                y = self.fabric.to_device(
                    batch["labels"]
                )  # labels are not used for compute the entropy loss, but just compute the accuracy
                correct = (logits.argmax(dim=-1) == y).sum().item()
                total_correct += correct
                total_count += y.size(0)
                self.fabric.log_dict(
                    {
                        f"loss/{task_name}": loss.item(),
                        f"accuracy/{task_name}": correct / y.size(0),
                    },
                    step=step_idx,
                )

            # update model parameters
            optimizer.zero_grad()
            self.fabric.backward(total_loss)
            optimizer.step()

            # log the loss and accuracy
            metrics = {
                "loss": total_loss.item(),
                "accuracy": total_correct / total_count,
            }
            self.fabric.log_dict(metrics, step=step_idx)
            pabr.set_postfix(metrics)
            if step_idx % cfg.save_interval == 0:
                ckpt_dir = self.result_dir / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                self.fabric.save(
                    ckpt_dir / f"model_{step_idx}.ckpt",
                    {"model": model},
                )

    def cleanup(self):
        log.info("Exiting")

    def load_datasets(self):
        "load the datasets by the given task name"
        from src.datasets.registry import get_dataset

        cfg = self.cfg

        # Load the datasets
        datasets = {
            task_name: get_dataset(
                task_name,
                cast(ImageEncoder, self.model).val_preprocess,
                location=cfg.data_location,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
            )
            for task_name in cfg.test_tasks
        }
        test_loaders = {
            dataset_name: DataLoader(
                dataset.test_dataset,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
            )
            for dataset_name, dataset in datasets.items()
        }
        # shuffled, drop_last=True
        infty_test_loaders = {
            dataset_name: itertools.cycle(
                DataLoader(
                    dataset.test_dataset,
                    batch_size=cfg.batch_size,
                    num_workers=cfg.num_workers,
                    shuffle=True,
                    pin_memory=True,
                    drop_last=True,
                )
            )
            for dataset_name, dataset in datasets.items()
        }

        self.datasets = datasets
        self.test_loaders = test_loaders
        self.infty_test_loaders = infty_test_loaders

    def load_model(self):
        """
        load a model from the checkpoint, this model is obtained by average the finetuned models
        """
        cfg = self.cfg

        with timeit_context():
            log.info("load models")
            # we don't need to load the pretrained model
            # pretrained_model = torch.load(
            #     pretrained_model_path(cfg.model), map_location="cpu"
            # )
            finetuned_models = [
                torch.load(
                    finetuned_model_path(cfg.model, task_name), map_location="cpu"
                )
                for task_name in track(cfg.tasks, "loading finetuned models")
            ]

        # construct the average model
        model = simple_average(finetuned_models)
        classification_heads = {
            task_name: get_classification_head(cfg, task_name)
            for task_name in cfg.tasks
        }

        self.model = self.fabric.setup_module(model)
        self.classification_heads = {
            task_name: self.fabric.setup_module(head)
            for task_name, head in classification_heads.items()
        }


@hydra.main(
    config_path=str(CONFIG_DIR), config_name="clip_entropy_tta", version_base=None
)
def main(cfg: DictConfig) -> None:
    (program := Program(cfg)).run()


if __name__ == "__main__":
    main()
