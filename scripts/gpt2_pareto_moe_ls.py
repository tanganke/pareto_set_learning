from _common import *

log = logging.getLogger(__name__)

from gpt2_pareto_moe import GPT2ParetoMoEProgram
from torch.nn.modules import Module

from scripts._common import DictConfig, Tensor, Tuple


class Program(GPT2ParetoMoEProgram):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg, __file__)

    def compute_loss(self, model: Module, ray: Tensor, losses: Tuple[Tensor]):
        loss = 0
        for r, l in zip(ray, losses):
            loss = loss + r * l
        return loss


@hydra.main(
    config_path=str(CONFIG_DIR),
    config_name="gpt2_pareto_moe",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    program = Program(cfg)
    program.run()


if __name__ == "__main__":
    main()
