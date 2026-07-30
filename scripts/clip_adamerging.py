R"""
Layer-wise AdaMerging (Yang et al., ICLR 2024).

The merging coefficients are learned per task **and** per parameter tensor by
minimizing the Shannon entropy of the predictions on the *unlabeled* test data:

.. math::
    \theta = \theta_0 + \sum_k \sum_l \lambda_k^l V_k^l ,
    \qquad \min_{\lambda} \sum_k H(f(x_k; \theta)) .

This is a strong single-model merging baseline (it produces one "one-size-fits-all"
model, without preference control) and also the base model on top of which
Pareto Merging is built in the original paper.

Usage::

    python scripts/clip_adamerging.py model=ViT-B-32 \
        test_datasets="[SUN397,Cars,RESISC45,EuroSAT,SVHN,GTSRB,MNIST,DTD]" version=0
"""

from _common import *

log = logging.getLogger(__name__)

from collections import defaultdict
from typing import Optional, cast

from clip_checkpoint_path import (
    CHECKPOINT_DIR,
    finetuned_model_path,
    pretrained_model_path,
)
from src.datasets.common import maybe_dictionarize
from src.heads import get_classification_head
from src.modeling import ImageEncoder
from src.utils import timeit_context
from torch.func import functional_call
from torch.utils.data import DataLoader


def entropy_loss(logits: Tensor) -> Tensor:
    """Mean Shannon entropy of the predicted distribution."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -torch.mean(torch.sum(probs * log_probs, dim=-1))


class LayerWiseMergedModel(nn.Module):
    """Pretrained model plus per-(task, parameter-tensor) scaled task vectors."""

    def __init__(
        self,
        pretrained_model: nn.Module,
        task_vectors: List[Dict[str, Tensor]],
        init_lambda: float = 0.3,
        clamp_weights: bool = False,
    ):
        super().__init__()
        # keep the module out of the parameter registry: only lambda is trainable
        self._model = (pretrained_model.requires_grad_(False),)
        self.pretrained_state_dict = {
            k: v.detach() for k, v in pretrained_model.state_dict().items()
        }
        self.param_names = list(self.pretrained_state_dict.keys())
        # stack the task vectors per parameter as a (K, numel) matrix so that the
        # merge is a single matmul: this keeps the autograd graph to one tensor per
        # parameter instead of one per (task, parameter) pair, which matters a lot
        # for ViT-L/14.
        self.stacked_task_vectors = {
            name: torch.stack([tv[name].reshape(-1) for tv in task_vectors])
            for name in self.param_names
        }
        self.clamp_weights = clamp_weights

        self.num_tasks = len(task_vectors)
        self.merge_weight = nn.Parameter(
            torch.full(
                (self.num_tasks, len(self.param_names)),
                init_lambda,
                dtype=torch.float32,
            ),
            requires_grad=True,
        )

    @property
    def model(self):
        return self._model[0]

    def to_device(self, device):
        self._model = (self.model.to(device),)
        self.pretrained_state_dict = {
            k: v.to(device) for k, v in self.pretrained_state_dict.items()
        }
        self.stacked_task_vectors = {
            k: v.to(device) for k, v in self.stacked_task_vectors.items()
        }
        return self.to(device)

    def merged_state_dict(self) -> Dict[str, Tensor]:
        weight = (
            self.merge_weight.clamp(0, 1) if self.clamp_weights else self.merge_weight
        )
        merged = {}
        for param_idx, name in enumerate(self.param_names):
            base = self.pretrained_state_dict[name]
            task_vectors = self.stacked_task_vectors[name]
            delta = weight[:, param_idx] @ task_vectors
            merged[name] = base + delta.view(base.shape)
        return merged

    def forward(self, *args, state_dict: Optional[Dict[str, Tensor]] = None, **kwargs):
        if state_dict is None:
            state_dict = self.merged_state_dict()
        return functional_call(self.model, state_dict, args=args, kwargs=kwargs)

    @torch.no_grad()
    def unload(self) -> nn.Module:
        """Return a plain module holding the merged weights."""
        model = deepcopy(self.model)
        model.load_state_dict({k: v.detach() for k, v in self.merged_state_dict().items()})
        return model


class Program:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        cfg.save = str(CHECKPOINT_DIR / cfg.model)
        cfg.data_location = str(DATA_DIR)
        self.result_dir = RESULTS_DIR / "clip_adamerging" / cfg.model
        if cfg.version is not None:
            self.result_dir = self.result_dir / f"version_{cfg.version}"
        self.result_dir.mkdir(parents=True, exist_ok=True)
        cfg.result_dir = str(self.result_dir)
        self.device = torch.device(cfg.device)

    # ------------------------------------------------------------------ loading
    def load_models(self):
        cfg = self.cfg
        with timeit_context("loading checkpoints"):
            pretrained_model: ImageEncoder = torch.load(
                pretrained_model_path(cfg.model), map_location="cpu"
            )
            task_vectors = []
            for dataset_name in track(cfg.test_datasets, "loading finetuned models"):
                finetuned: ImageEncoder = torch.load(
                    finetuned_model_path(cfg.model, dataset_name), map_location="cpu"
                )
                pretrained_sd = pretrained_model.state_dict()
                task_vectors.append(
                    {
                        k: (v - pretrained_sd[k]).detach()
                        for k, v in finetuned.state_dict().items()
                    }
                )
                del finetuned

        self.pretrained_model = pretrained_model
        self.classification_heads = {
            name: get_classification_head(cfg, name)
            .requires_grad_(False)
            .eval()
            .to(self.device)
            for name in cfg.test_datasets
        }
        self.merged_model = LayerWiseMergedModel(
            pretrained_model=deepcopy(pretrained_model),
            task_vectors=task_vectors,
            init_lambda=cfg.init_lambda,
            clamp_weights=cfg.clamp_weights,
        ).to_device(self.device)

    def load_datasets(self):
        cfg = self.cfg
        from src.datasets.registry import get_dataset

        datasets = [
            get_dataset(
                name,
                self.pretrained_model.val_preprocess,
                location=cfg.data_location,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
            )
            for name in cfg.test_datasets
        ]
        self.test_datasets = [d.test_dataset for d in datasets]
        self.test_loaders = [
            DataLoader(
                d,
                shuffle=False,
                batch_size=cfg.eval_batch_size,
                num_workers=cfg.num_workers,
            )
            for d in self.test_datasets
        ]
        # unlabeled test-time adaptation stream
        self.tta_loader_iters = [
            iter(
                itertools.cycle(
                    DataLoader(
                        d,
                        shuffle=True,
                        batch_size=cfg.tta_batch_size,
                        num_workers=cfg.num_workers,
                    )
                )
            )
            for d in self.test_datasets
        ]

    # ----------------------------------------------------------------- training
    def test_time_adaptation(self):
        cfg = self.cfg
        optimizer = torch.optim.Adam([self.merged_model.merge_weight], lr=cfg.lr)

        num_tasks = len(cfg.test_datasets)
        for step_idx in tqdm(
            range(1, 1 + cfg.num_steps), "AdaMerging TTA", dynamic_ncols=True
        ):
            optimizer.zero_grad()
            # merge once per step and back-propagate task by task, so the activations
            # of one task are freed before the next forward pass
            state_dict = self.merged_model.merged_state_dict()
            total_loss = 0.0
            for dataset_idx, dataset_name in enumerate(cfg.test_datasets):
                batch = maybe_dictionarize(next(self.tta_loader_iters[dataset_idx]))
                x = batch["images"].to(self.device)
                features = self.merged_model(x, state_dict=state_dict)
                logits = self.classification_heads[dataset_name](features)
                loss = entropy_loss(logits)
                loss.backward(retain_graph=dataset_idx < num_tasks - 1)
                total_loss += loss.item()
            optimizer.step()

            if step_idx % 50 == 0:
                log.info(f"step {step_idx}: entropy loss {total_loss:.4f}")

        torch.save(
            self.merged_model.merge_weight.detach().cpu(),
            self.result_dir / "merge_weight.pt",
        )

    # --------------------------------------------------------------- evaluation
    @torch.inference_mode()
    def evaluate(self):
        cfg = self.cfg
        results = defaultdict(list)
        accs = []
        # merge once, then reuse the merged weights for every batch
        state_dict = self.merged_model.merged_state_dict()
        for dataset_idx, dataset_name in enumerate(cfg.test_datasets):
            correct = total = 0
            for batch in tqdm(
                self.test_loaders[dataset_idx], f"evaluate {dataset_name}", leave=False
            ):
                batch = maybe_dictionarize(batch)
                x = batch["images"].to(self.device)
                y = batch["labels"].to(self.device)
                logits = self.classification_heads[dataset_name](
                    self.merged_model(x, state_dict=state_dict)
                )
                correct += (logits.argmax(-1) == y).sum().item()
                total += len(y)
            acc = correct / total
            accs.append(acc)
            results[dataset_name].append(acc)
            log.info(f"{dataset_name}: {100 * acc:.2f}%")

        results["Avg."].append(float(np.mean(accs)))
        df = pd.DataFrame(results)
        df.to_csv(self.result_dir / "result.csv", index=False)
        log.info(f"\n{df}")
        log.info(f"average accuracy: {100 * np.mean(accs):.2f}%")

    def run(self):
        self.load_models()
        self.load_datasets()
        if self.cfg.train:
            self.test_time_adaptation()
        if self.cfg.evaluate:
            self.evaluate()


@hydra.main(
    config_path=str(CONFIG_DIR),
    config_name="clip_adamerging",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    Program(cfg).run()


if __name__ == "__main__":
    main()
