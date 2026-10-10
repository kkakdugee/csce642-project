import unittest

import numpy as np

from navigation.envs.room import RoomEnv
from navigation.place_cells import PlaceCells
from navigation.trajectories import generate_batch


class TrajectoryTests(unittest.TestCase):
    def test_batch_shapes_types_and_normalization(self):
        for dimensions in (2, 3):
            encoder = PlaceCells.sample(
                n_cells=5,
                bounds=[1] * dimensions,
                sigma=0.3,
                rng=np.random.default_rng(7),
            )
            for batch_size, sequence_length in ((1, 1), (1, 3), (2, 1)):
                with self.subTest(
                    dimensions=dimensions, batch=batch_size, length=sequence_length
                ):
                    batch = generate_batch(
                        grid_shape=(3,) * dimensions,
                        step_size=0.5,
                        place_cells=encoder,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        rng=np.random.default_rng(8),
                    )
                    self.assertEqual(
                        set(batch), {"initial_code", "displacements", "targets"}
                    )
                    self.assertEqual(batch["initial_code"].shape, (batch_size, 5))
                    self.assertEqual(
                        batch["displacements"].shape,
                        (batch_size, sequence_length, dimensions),
                    )
                    self.assertEqual(
                        batch["targets"].shape, (batch_size, sequence_length, 5)
                    )
                    for array in batch.values():
                        self.assertEqual(array.dtype, np.float32)
                        self.assertTrue(np.isfinite(array).all())
                    for name in ("initial_code", "targets"):
                        self.assertTrue(np.all(batch[name] >= 0))
                        np.testing.assert_allclose(
                            batch[name].sum(axis=-1), 1.0, rtol=1e-6
                        )

    def test_reproducibility_without_mutating_global_rng_or_encoder(self):
        before = np.random.get_state()
        encoder = PlaceCells.sample(
            n_cells=5, bounds=[2, 2], sigma=0.5, rng=np.random.default_rng(13)
        )
        centers_before = encoder.centers.copy()
        parameters = dict(
            grid_shape=(5, 5),
            step_size=0.5,
            place_cells=encoder,
            batch_size=3,
            sequence_length=5,
        )
        first = generate_batch(**parameters, rng=np.random.default_rng(17))
        generate_batch(**parameters, rng=np.random.default_rng(29))
        second = generate_batch(**parameters, rng=np.random.default_rng(17))
        after = np.random.get_state()
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        np.testing.assert_array_equal(encoder.centers, centers_before)
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_physical_displacements_and_target_alignment_match_room(self):
        for shape in ((3, 4), (3, 4, 2)):
            # Put an encoder center at every grid point to recover positions.
            grid_cells = np.array(list(np.ndindex(shape)))
            centers = grid_cells * 0.25
            encoder = PlaceCells(centers=centers, sigma=0.2)
            batch = generate_batch(
                grid_shape=shape,
                step_size=0.25,
                place_cells=encoder,
                batch_size=8,
                sequence_length=5,
                rng=np.random.default_rng(11),
            )
            initial_codes = batch["initial_code"][:, None, :]
            all_codes = np.concatenate((initial_codes, batch["targets"]), axis=1)
            center_indices = all_codes.argmax(axis=-1)
            cells = grid_cells[center_indices]
            positions = centers[center_indices]

            # Each target must describe the position after its displacement.
            expected_codes = encoder.encode(positions)
            expected_displacements = np.diff(positions, axis=1)
            displacements = batch["displacements"]
            valid_components = np.isin(displacements, [-0.25, 0, 0.25])
            moving_axis_count = np.count_nonzero(displacements, axis=-1)
            blocked_moves = np.all(displacements == 0, axis=-1)
            np.testing.assert_allclose(all_codes, expected_codes, rtol=1e-6, atol=1e-7)
            np.testing.assert_array_equal(expected_displacements, displacements)
            self.assertTrue(valid_components.all())
            self.assertTrue(np.all(moving_axis_count <= 1))
            self.assertTrue(np.any(blocked_moves))

            # Replay each sequence with a goal that the sequence never visits.
            last_cell = np.asarray(shape) - 1
            for row in range(len(cells)):
                visited_indices = set(center_indices[row])
                goal_index = next(
                    index
                    for index in range(len(grid_cells))
                    if index not in visited_indices
                )
                room = RoomEnv(
                    grid_shape=shape,
                    step_size=0.25,
                    goal_radius=0.0625,
                    max_episode_steps=6,
                )
                room.reset(
                    options={
                        "start_cell": cells[row, 0],
                        "goal_cell": grid_cells[goal_index],
                    }
                )
                for time, displacement in enumerate(displacements[row]):
                    current_cell = cells[row, time]
                    moved_axes = np.flatnonzero(displacement)
                    if len(moved_axes):
                        axis = int(moved_axes[0])
                        direction = int(displacement[axis] < 0)
                    else:
                        boundary_mask = (current_cell == 0) | (
                            current_cell == last_cell
                        )
                        boundary_axes = np.flatnonzero(boundary_mask)
                        self.assertGreater(len(boundary_axes), 0)
                        axis = int(boundary_axes[0])
                        direction = int(current_cell[axis] == 0)
                    action = 2 * axis + direction
                    observation, _, terminated, truncated, info = room.step(action)
                    np.testing.assert_array_equal(
                        observation["displacement"], displacement
                    )
                    np.testing.assert_array_equal(
                        info["position"], positions[row, time + 1]
                    )
                    self.assertFalse(terminated or truncated)

    def test_invalid_batch_inputs(self):
        encoder = PlaceCells(centers=[[0, 0], [1, 1]], sigma=0.5)
        invalid = (
            {"grid_shape": (3,)},
            {"grid_shape": (1, 3)},
            {"grid_shape": (3, 3, 3)},
            {"step_size": 0},
            {"batch_size": 0},
            {"sequence_length": 0},
            {"sequence_length": 1.5},
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                parameters = dict(
                    grid_shape=(3, 3),
                    step_size=0.5,
                    place_cells=encoder,
                    batch_size=2,
                    sequence_length=3,
                )
                parameters.update(arguments)
                with self.assertRaises((ValueError, TypeError)):
                    generate_batch(**parameters, rng=np.random.default_rng(9))


if __name__ == "__main__":
    unittest.main()
