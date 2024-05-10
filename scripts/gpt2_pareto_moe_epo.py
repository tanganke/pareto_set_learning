from _common import *

log = logging.getLogger(__name__)

from gpt2_pareto_moe import GPT2ParetoMoEProgram
from torch.nn.modules import Module

from scripts._common import DictConfig, Tensor, Tuple
from src.phn.solvers import EPOSolver


class Program(GPT2ParetoMoEProgram):
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
    config_name="gpt2_pareto_moe",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    program = Program(cfg)
    program.run()


if __name__ == "__main__":
    main()
