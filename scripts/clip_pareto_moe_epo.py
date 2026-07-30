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
from lightning.pytorch.callbacks import (
    RichModelSummary,
    DeviceStatsMonitor,
    LearningRateMonitor,
)

from clip_pareto_moe import CLIPParetoMoEProgram


class Program(CLIPParetoMoEProgram):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg, __file__)

        self.epo_solver = None

    def compute_loss(self, model: nn.Module, ray: Tensor, losses: Tuple[Tensor]):
        if self.epo_solver is None:
            num_objectives = len(self.finetuned_models)
            self.epo_solver = EPOSolver(n_tasks=num_objectives, n_params=None)
        epo_solver = self.epo_solver

        losses = torch.stack(losses)
        loss = epo_solver.get_weighted_loss(
            losses,
            ray,
            tuple(filter(lambda p: p.requires_grad, model.parameters())),
        )
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
