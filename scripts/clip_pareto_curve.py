R"""
Given two well-trained models on two tasks, we want to find the Pareto set from them.
This Pareto set is expected to be a spline curve that connects these two models.

dataset can be one of:
  - SUN397
  - Cars
  - RESISC45
  - EuroSAT
  - SVHN
  - GTSRB
  - MNIST
  - DTD
"""

from rich.logging import RichHandler

from _common import *

logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)

import wandb
from src.utils import timeit_context, first
from src.modeling import ImageEncoder
from clip_checkpoint_path import CHECKPOINT_DIR


class Progam:
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        cfg.save = str(CHECKPOINT_DIR / cfg.model)
        cfg.data_location = str(DATA_DIR)

        wandb.init(
            project="pareto_manifold",
            config=OmegaConf.to_container(cfg),
            name=__name__,
        )

    def main(self):
        self.load_models()
        self.load_datasets()

    def load_models(self):
        "load two well-trained models by the given task name"
        from src.heads import get_classification_head

        from clip_checkpoint_path import finetuned_model_path

        cfg = self.cfg

        with timeit_context("loading models"):
            models = {}
            for task in cfg.tasks:
                model_path = finetuned_model_path(cfg.model, task)
                model = torch.load(model_path, map_location="cpu")
                models[task] = model

            heads = {}
            for task in cfg.tasks:
                heads[task] = get_classification_head(cfg, task)

        self.models = models
        self.heads = heads

    def load_datasets(self):
        "load the datasets by the given task name"
        from src.datasets.registry import get_dataset

        cfg = self.cfg

        # Load the datasets
        datasets = {
            dataset_name: get_dataset(
                dataset_name,
                cast(ImageEncoder, first(self.models.values())).val_preprocess,
                location=cfg.data_location,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
            )
            for dataset_name in cfg.tasks
        }
        train_loaders = {
            dataset_name: dataset.train_loader
            for dataset_name, dataset in datasets.items()
        }
        train_loaders_infty = {
            dataset_name: itertools.cycle(dataloader)
            for dataset_name, dataloader in train_loaders.items()
        }

        self.datasets = datasets
        self.train_loaders = train_loaders
        self.train_loaders_infty = train_loaders_infty


@hydra.main(
    config_path=str(CONFIG_DIR),
    config_name="clip_pareto_curve",
    version_base=None,
)
def main(cfg: DictConfig):
    program = Progam(cfg)
    program.main()


if __name__ == "__main__":
    main()
