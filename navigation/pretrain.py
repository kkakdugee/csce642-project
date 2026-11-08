"""Pretrain and validate one position network on CPU."""

import argparse
import json
import math
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from navigation.place_cells import PlaceCells
from navigation.position_network import PositionNetwork
from navigation.trajectories import generate_batch


@dataclass(frozen=True)
class PretrainingConfig:
    grid_shape: tuple[int, ...]
    step_size: float = 1.0
    n_cells: int = 32
    sigma: float = 1.0
    hidden_size: int = 64
    batch_size: int = 16
    sequence_length: int = 16
    train_batches: int = 200
    validation_batches: int = 16
    learning_rate: float = 0.003
    seed: int = 0

    def build_model(self) -> PositionNetwork:
        # Seed weight initialization independently of the caller's CPU RNG.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(self.seed)
            return PositionNetwork(
                dimensions=len(self.grid_shape),
                n_cells=self.n_cells,
                hidden_size=self.hidden_size,
            )

    def batch(self, cells: PlaceCells, rng: np.random.Generator, *, positions=False):
        return generate_batch(
            grid_shape=self.grid_shape,
            step_size=self.step_size,
            place_cells=cells,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            rng=rng,
            include_positions=positions,
        )


def _loss_and_log_probs(model, batch, *, include_cue):
    cue = torch.from_numpy(batch["initial_code"])
    moves = torch.from_numpy(batch["displacements"])
    targets = torch.from_numpy(batch["targets"])
    state = model.initial_state(cue)
    activity = model.sequence_activity(moves, state)
    if include_cue:
        # Navigation also reads the starting or goal cue before the first move.
        activity = torch.cat((state[:, None], activity), dim=1)
        targets = torch.cat((cue[:, None], targets), dim=1)

    log_probs = model.readout(activity).log_softmax(dim=-1)
    loss = -(targets * log_probs).sum(dim=-1).mean()
    return loss, log_probs


@torch.no_grad()
def _validate(model, config, cells, rng):
    """Score post-move predictions against soft targets and true room coordinates."""
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    for _ in range(config.validation_batches):
        batch = config.batch(cells, rng, positions=True)
        loss, log_probs = _loss_and_log_probs(model, batch, include_cue=False)
        # Decode a position as the probability-weighted mean of the fixed centers.
        decoded = log_probs.exp().numpy() @ cells.centers
        errors = np.linalg.norm(decoded - batch["positions"], axis=-1)
        total_loss += loss.item()
        total_error += float(errors.mean())

    return (
        total_loss / config.validation_batches,
        total_error / config.validation_batches,
    )


def _save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Finish writing before replacing an existing checkpoint.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path):
    """Restore the CPU model in eval mode, its place-cell population and run config."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    config = PretrainingConfig(**saved["config"])
    model = config.build_model()
    model.load_state_dict(saved["model_state"])
    cells = PlaceCells(centers=saved["centers"].numpy(), sigma=config.sigma)
    return model.eval(), cells, config


def pretrain(config: PretrainingConfig, path: Path) -> dict:
    """Train on fresh walks, validate on independent walks, and save one checkpoint."""
    if min(config.train_batches, config.validation_batches) < 1:
        raise ValueError("train_batches and validation_batches must be positive.")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")

    # The validation walks stay fixed when the training budget changes.
    seeds = np.random.SeedSequence(config.seed).spawn(3)
    center_rng, train_rng, validation_rng = map(np.random.default_rng, seeds)
    cells = PlaceCells.sample(
        n_cells=config.n_cells,
        bounds=(np.asarray(config.grid_shape) - 1) * config.step_size,
        sigma=config.sigma,
        rng=center_rng,
    )
    model = config.build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    started = perf_counter()
    for _ in range(config.train_batches):
        batch = config.batch(cells, train_rng)
        optimizer.zero_grad()
        loss, _ = _loss_and_log_probs(model, batch, include_cue=True)
        loss.backward()
        optimizer.step()

    trained = perf_counter()
    loss, error = _validate(model, config, cells, validation_rng)
    if not (math.isfinite(loss) and math.isfinite(error)):
        raise FloatingPointError("Non-finite validation result.")

    steps_per_batch = config.batch_size * config.sequence_length
    metrics = {
        "dimensions": len(config.grid_shape),
        "validation_loss": loss,
        "mean_position_error": error,
        "train_steps": config.train_batches * steps_per_batch,
        "validation_steps": config.validation_batches * steps_per_batch,
        "training_seconds": trained - started,
        "validation_seconds": perf_counter() - trained,
        "device": "cpu",
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "torch_threads": torch.get_num_threads(),
    }
    _save_checkpoint(
        path,
        {
            "config": asdict(config),
            "model_state": model.state_dict(),
            "centers": torch.from_numpy(cells.centers.copy()),
            "optimizer": {"name": "Adam", **optimizer.defaults},
            "torch_version": str(torch.__version__),
            "metrics": metrics,
        },
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, argument_default=argparse.SUPPRESS
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-shape", type=int, nargs="+", required=True)
    parser.add_argument("--step-size", type=float)
    parser.add_argument("--n-cells", type=int)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--train-batches", type=int)
    parser.add_argument("--validation-batches", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    settings = vars(parser.parse_args())
    output = settings.pop("output")
    settings["grid_shape"] = tuple(settings["grid_shape"])
    metrics = pretrain(PretrainingConfig(**settings), output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
