import unittest

import numpy as np
import torch

from navigation.place_cells import PlaceCells
from navigation.position_network import PositionNetwork
from navigation.trajectories import generate_batch


class PositionNetworkTests(unittest.TestCase):
    def test_step_matches_trajectory_batch(self):
        for dimensions in (2, 3):
            rng = np.random.default_rng(13)
            cells = PlaceCells.sample(
                n_cells=5, bounds=[2] * dimensions, sigma=1, rng=rng
            )
            model = PositionNetwork(dimensions=dimensions, n_cells=5, hidden_size=7)
            for size, length in ((1, 1), (1, 4), (3, 1), (3, 4)):
                with self.subTest(dimensions=dimensions, batch=size, time=length):
                    batch = generate_batch(
                        grid_shape=(3,) * dimensions,
                        step_size=1,
                        place_cells=cells,
                        batch_size=size,
                        sequence_length=length,
                        rng=rng,
                    )
                    cue = torch.from_numpy(batch["initial_code"])
                    moves = torch.from_numpy(batch["displacements"])
                    activity, predictions = model(cue, moves)
                    self.assertEqual(activity.shape, (size, length, 7))
                    self.assertEqual(predictions.shape, batch["targets"].shape)

                    # Each returned feature vector must be the input to the readout.
                    expected = model.readout(activity).softmax(dim=-1)
                    torch.testing.assert_close(predictions, expected)
                    torch.testing.assert_close(
                        predictions.sum(-1), torch.ones(size, length)
                    )

                    state = model.initial_state(cue)
                    for t in range(length):
                        state, prediction = model.step(moves[:, t], state)
                        torch.testing.assert_close(state, activity[:, t])
                        torch.testing.assert_close(prediction, predictions[:, t])

    def test_new_cue_resets_memory_before_any_move(self):
        model = PositionNetwork(dimensions=2, n_cells=3, hidden_size=5)
        start, goal = torch.eye(3)[:2].split(1)
        move = torch.tensor([[1.0, 0.0]])
        expected = model.step(move, model.initial_state(start))

        # Decode the goal at reset, then follow a different movement history.
        goal_state = model.initial_state(goal)
        goal_prediction = model.predict_place_cells(goal_state)
        self.assertEqual(goal_prediction.shape, (1, 3))
        self.assertFalse(torch.allclose(goal_state, model.initial_state(start)))
        state, _ = model.step(-move, goal_state)
        after_walk, _ = model.step(move, state)

        restarted = model.step(move, model.initial_state(start))
        self.assertFalse(torch.allclose(restarted[0], after_walk))
        torch.testing.assert_close(restarted, expected)

    def test_rejects_unsupported_dimensions(self):
        with self.assertRaises(ValueError):
            PositionNetwork(dimensions=4, n_cells=3, hidden_size=5)
