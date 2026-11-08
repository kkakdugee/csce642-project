import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from navigation.place_cells import PlaceCells
from navigation.pretrain import PretrainingConfig, load_checkpoint, pretrain
from navigation.trajectories import generate_batch


class PretrainingTests(unittest.TestCase):
    def test_validation_positions_align_with_cues_moves_and_targets(self):
        cells = PlaceCells(centers=[[0, 0], [0.9, 0.1], [0.1, 0.9]], sigma=0.4)
        batch = generate_batch(
            grid_shape=(5, 5),
            step_size=0.25,
            place_cells=cells,
            batch_size=3,
            sequence_length=6,
            rng=np.random.default_rng(7),
            include_positions=True,
        )
        positions = batch["positions"]
        moves = batch["displacements"]
        start = positions[:, 0] - moves[:, 0]
        np.testing.assert_allclose(cells.encode(start), batch["initial_code"])
        np.testing.assert_array_equal(np.diff(positions, axis=1), moves[:, 1:])
        np.testing.assert_allclose(cells.encode(positions), batch["targets"])

    def test_reloaded_checkpoint_reproduces_held_out_metrics(self):
        for dimensions in (2, 3):
            with (
                self.subTest(dimensions=dimensions),
                tempfile.TemporaryDirectory() as tmp,
            ):
                config = PretrainingConfig(
                    grid_shape=(3,) * dimensions,
                    step_size=0.5,
                    n_cells=5,
                    sigma=0.3,
                    hidden_size=7,
                    batch_size=2,
                    sequence_length=3,
                    train_batches=2,
                    validation_batches=3,
                    learning_rate=0.01,
                    seed=17,
                )
                path = Path(tmp) / "checkpoint.pt"
                metrics = pretrain(config, path)
                model, cells, loaded_config = load_checkpoint(path)
                self.assertEqual(loaded_config, config)
                self.assertEqual(cells.sigma, config.sigma)
                initial_weights = config.build_model().readout.weight
                self.assertFalse(torch.equal(model.readout.weight, initial_weights))
                self.assertEqual(metrics["train_steps"], 12)
                self.assertEqual(metrics["validation_steps"], 18)

                # Replay the held-out walks independently of the trainer.
                center_seed, _, validation_seed = np.random.SeedSequence(17).spawn(3)
                expected_cells = PlaceCells.sample(
                    n_cells=5,
                    bounds=[1] * dimensions,
                    sigma=0.3,
                    rng=np.random.default_rng(center_seed),
                )
                np.testing.assert_array_equal(cells.centers, expected_cells.centers)
                rng = np.random.default_rng(validation_seed)
                losses, errors = [], []
                for _ in range(config.validation_batches):
                    batch = generate_batch(
                        grid_shape=config.grid_shape,
                        step_size=config.step_size,
                        place_cells=expected_cells,
                        batch_size=config.batch_size,
                        sequence_length=config.sequence_length,
                        rng=rng,
                        include_positions=True,
                    )
                    cue = torch.from_numpy(batch["initial_code"])
                    moves = torch.from_numpy(batch["displacements"])
                    with torch.no_grad():
                        _, prediction = model(cue, moves)
                    probabilities = prediction.numpy()
                    decoded = probabilities @ cells.centers
                    errors.append(np.linalg.norm(decoded - batch["positions"], axis=-1))
                    losses.append(-(batch["targets"] * np.log(probabilities)).sum(-1))

                self.assertAlmostEqual(metrics["validation_loss"], np.mean(losses), 5)
                self.assertAlmostEqual(
                    metrics["mean_position_error"], np.mean(errors), 5
                )
