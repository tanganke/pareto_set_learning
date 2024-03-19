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

from src.clip_eval import eval_single_dataset, eval_single_dataset_preprocess_head
from src.heads import get_classification_head
from src.modeling import ClassificationHead, ImageEncoder
from src.utils import first, timeit_context


class Program:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        cfg.save = str(CHECKPOINT_DIR / cfg.model)
        cfg.data_location = str(DATA_DIR)

        self.fabric = L.Fabric(accelerator="cuda", devices=1)
        self.fabric.launch()

    def run(self):
        self.load_model()

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
            for task_name in cfg.tasks
        }
        train_loaders = {
            dataset_name: dataset.train_loader
            for dataset_name, dataset in datasets.items()
        }
        train_loaders_infty = {
            dataset_name: itertools.cycle(dataloader)
            for dataset_name, dataloader in train_loaders.items()
        }

        test_loaders = {
            dataset_name: dataset.test_loader
            for dataset_name, dataset in datasets.items()
        }

        self.datasets = datasets
        self.train_loaders = train_loaders
        self.train_loaders_infty = train_loaders_infty
        self.test_loaders = test_loaders

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


@hydra.main(config_path=str(CONFIG_DIR), config_name="clip_default", version_base=None)
def main(cfg: DictConfig) -> None:
    (program := Program(cfg)).run()


if __name__ == "__main__":
    main()
