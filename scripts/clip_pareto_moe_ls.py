from torch.nn.modules import Module
from _common import *
from scripts._common import Tensor

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
from torch.utils.data import DataLoader

from src.clip_eval import eval_single_dataset
from src.datasets.common import maybe_dictionarize
from src.heads import get_classification_head
from src.modeling import ClassificationHead, ImageEncoder
from src.module.dict_moe import ParetoWeightEnsemblingModule
from src.module.utils import get_by_name, print_trainable_parameters, set_by_name
from src.task_vectors import StateDict, TaskVector, state_dict_mean
from src.utils import timeit_context
from lightning.pytorch.loggers import TensorBoardLogger
from open_clip.model import ResidualAttentionBlock, CLIP, VisualTransformer
from torch.utils.data.distributed import DistributedSampler
from lightning.pytorch.callbacks import (
    RichModelSummary,
    DeviceStatsMonitor,
    LearningRateMonitor,
)
from clip_pareto_moe import CLIPParetoMoEProgram


class Program(CLIPParetoMoEProgram):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg, __file__)

    def compute_loss(
        self, model: Module, ray: Tensor, losses:Tuple[Tensor]
    ):
        loss = 0
        for r, l in zip(ray, losses):
            loss += r * l
        return loss


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
