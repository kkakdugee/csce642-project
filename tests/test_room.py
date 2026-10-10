import unittest

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from navigation.envs.movement import move_cell
from navigation.envs.room import RoomEnv


class RoomEnvTests(unittest.TestCase):
    def test_gymnasium_contract_in_both_dimensions(self):
        for dimensions in (2, 3):
            with self.subTest(dimensions=dimensions):
                env = RoomEnv(grid_shape=(5,) * dimensions)
                check_env(env, skip_render_check=True)
                observation, info = env.reset(seed=19)
                self.assertEqual(set(observation), {"displacement", "previous_action"})
                self.assertEqual(observation["displacement"].dtype, np.float32)
                np.testing.assert_array_equal(
                    observation["displacement"], np.zeros(dimensions)
                )
                self.assertEqual(observation["previous_action"], 2 * dimensions)
                self.assertTrue(env.observation_space.contains(observation))
                self.assertEqual(
                    set(info), {"position", "goal", "elapsed_steps", "is_success"}
                )
                self.assertEqual(info["position"].dtype, np.float64)
                self.assertEqual(info["goal"].dtype, np.float64)
                self.assertEqual(info["elapsed_steps"], 0)
                self.assertFalse(info["is_success"])

    def test_all_axis_actions_and_physical_step_size(self):
        moves = (
            (0.5, 0, 0),
            (-0.5, 0, 0),
            (0, 0.5, 0),
            (0, -0.5, 0),
            (0, 0, 0.5),
            (0, 0, -0.5),
        )
        for dimensions in (2, 3):
            env = RoomEnv(grid_shape=(5,) * dimensions, step_size=0.5)
            for action, move in enumerate(moves[: 2 * dimensions]):
                with self.subTest(dimensions=dimensions, action=action):
                    start = np.full(dimensions, 2, dtype=np.int64)
                    env.reset(
                        options={"start_cell": start, "goal_cell": [4] * dimensions}
                    )
                    observation, reward, terminated, truncated, info = env.step(action)
                    expected_position = start * 0.5 + move[:dimensions]
                    np.testing.assert_array_equal(
                        observation["displacement"], move[:dimensions]
                    )
                    np.testing.assert_array_equal(info["position"], expected_position)
                    self.assertEqual(observation["previous_action"], action)
                    self.assertEqual(reward, -0.01)
                    self.assertFalse(terminated or truncated)
                    self.assertEqual(info["elapsed_steps"], 1)

    def test_rejected_wall_move_counts_as_step_and_reports_zero_movement(self):
        env = RoomEnv(grid_shape=(5, 5))
        boundary_cases = (
            ([0, 0], [4, 4], 1),
            ([4, 4], [0, 0], 0),
        )
        for start, goal, action in boundary_cases:
            with self.subTest(action=action):
                env.reset(options={"start_cell": start, "goal_cell": goal})
                observation, reward, terminated, truncated, info = env.step(action)
                np.testing.assert_array_equal(observation["displacement"], [0, 0])
                np.testing.assert_array_equal(info["position"], start)
                self.assertEqual(observation["previous_action"], action)
                self.assertEqual(info["elapsed_steps"], 1)
                self.assertEqual(reward, -0.01)
                self.assertFalse(terminated or truncated)

    def test_movement_result_cannot_mutate_input(self):
        for action in (0, 1):
            with self.subTest(action=action):
                cell = np.array([0, 0], dtype=np.int64)
                result = move_cell(cell, action, (5, 5))
                result[:] = 3
                np.testing.assert_array_equal(cell, [0, 0])

    def test_success_on_last_step_takes_precedence_over_timeout(self):
        env = RoomEnv(
            grid_shape=(3, 3), max_episode_steps=2, goal_reward=2.0, step_penalty=0.1
        )
        env.reset(options={"start_cell": [0, 0], "goal_cell": [2, 0]})
        _, reward, terminated, truncated, _ = env.step(0)
        self.assertEqual(reward, -0.1)
        self.assertFalse(terminated or truncated)
        _, reward, terminated, truncated, info = env.step(0)
        self.assertAlmostEqual(reward, 1.9)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["is_success"])
        self.assertEqual(info["elapsed_steps"], 2)
        with self.assertRaises(gym.error.ResetNeeded):
            env.step(0)
        observation, info = env.reset(
            options={"start_cell": [0, 0], "goal_cell": [2, 2]}
        )
        self.assertEqual(info["elapsed_steps"], 0)
        self.assertFalse(info["is_success"])
        self.assertEqual(observation["previous_action"], 4)

    def test_timeout_and_step_before_reset(self):
        env = RoomEnv(grid_shape=(3, 3), max_episode_steps=1, step_penalty=0)
        with self.assertRaises(gym.error.ResetNeeded):
            env.step(0)
        env.reset(options={"start_cell": [0, 0], "goal_cell": [2, 2]})
        _, reward, terminated, truncated, info = env.step(0)
        self.assertEqual(reward, 0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertFalse(info["is_success"])
        with self.assertRaises(gym.error.ResetNeeded):
            env.step(0)

    def test_goal_radius_includes_its_boundary(self):
        env = RoomEnv(grid_shape=(3, 3), step_size=0.5, goal_radius=0.5)
        env.reset(options={"start_cell": [0, 0], "goal_cell": [2, 0]})
        _, _, terminated, truncated, _ = env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_decimal_goal_radius_is_consistent_at_reset_and_step(self):
        env = RoomEnv(grid_shape=(6, 2), step_size=0.1, goal_radius=0.3)
        env.reset(options={"start_cell": [0, 0], "goal_cell": [4, 0]})
        _, _, terminated, truncated, _ = env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        with self.assertRaises(ValueError):
            env.reset(options={"start_cell": [0, 0], "goal_cell": [3, 0]})

    def test_integer_endpoint_maps_to_decimal_physical_boundary(self):
        env = RoomEnv(grid_shape=(4, 2), step_size=0.3)
        _, info = env.reset(options={"start_cell": [0, 0], "goal_cell": [3, 0]})
        np.testing.assert_allclose(info["goal"], [0.9, 0])
        for _ in range(3):
            _, _, terminated, truncated, info = env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        np.testing.assert_allclose(info["position"], [0.9, 0])

    def test_seeded_reset_is_reproducible_and_samples_vary(self):
        first, second = RoomEnv(), RoomEnv()
        starts, goals = set(), set()
        for index in range(12):
            seed = 71 if index == 0 else None
            _, left = first.reset(seed=seed)
            _, right = second.reset(seed=seed)
            for name in ("position", "goal"):
                np.testing.assert_array_equal(left[name], right[name])
                np.testing.assert_array_equal(left[name], np.round(left[name]))
            self.assertGreater(np.linalg.norm(left["position"] - left["goal"]), 0.25)
            starts.add(tuple(left["position"]))
            goals.add(tuple(left["goal"]))
        self.assertGreater(len(starts), 1)
        self.assertGreater(len(goals), 1)

    def test_partial_reset_options_preserve_the_supplied_position(self):
        env = RoomEnv(
            grid_shape=(np.int64(3), np.int64(3)),
            step_size=0.5,
            max_episode_steps=np.int64(10),
        )
        for option, info_key in (("start_cell", "position"), ("goal_cell", "goal")):
            with self.subTest(option=option):
                _, info = env.reset(
                    seed=7, options={option: np.array([1, 2], dtype=np.int64)}
                )
                np.testing.assert_array_equal(info[info_key], [0.5, 1])
                self.assertGreater(
                    np.linalg.norm(info["position"] - info["goal"]), 0.25
                )

    def test_sampling_rejects_fixed_cell_with_no_separate_partner(self):
        env = RoomEnv(grid_shape=(3, 3), goal_radius=1.5)
        for option in ("start_cell", "goal_cell"):
            with self.subTest(option=option):
                with self.assertRaises(ValueError):
                    env.reset(seed=7, options={option: [1, 1]})

    def test_inputs_and_returned_arrays_cannot_mutate_simulator_state(self):
        env = RoomEnv(grid_shape=(5, 5))
        start = np.array([0, 0], dtype=np.int64)
        goal = np.array([4, 4], dtype=np.int64)
        observation, info = env.reset(options={"start_cell": start, "goal_cell": goal})
        start[:] = 3
        goal[:] = 3
        observation["displacement"][:] = 8
        info["position"][:] = 2
        info["goal"][:] = 2
        observation, _, _, _, info = env.step(0)
        np.testing.assert_array_equal(info["position"], [1, 0])
        np.testing.assert_array_equal(info["goal"], [4, 4])
        np.testing.assert_array_equal(observation["displacement"], [1, 0])
        observation["displacement"][:] = 8
        info["position"][:] = 2
        info["goal"][:] = 2
        observation, _, _, _, info = env.step(2)
        np.testing.assert_array_equal(info["position"], [1, 1])
        np.testing.assert_array_equal(info["goal"], [4, 4])
        np.testing.assert_array_equal(observation["displacement"], [0, 1])

    def test_invalid_configuration(self):
        invalid = (
            {"grid_shape": (5,)},
            {"grid_shape": (5, 5, 5, 5)},
            {"grid_shape": (1, 5)},
            {"grid_shape": (3.5, 5)},
            {"step_size": 0},
            {"step_size": np.inf},
            {"step_size": 1e-46},
            {"step_size": 1e39},
            {"goal_radius": 0},
            {"goal_radius": np.nan},
            {"goal_reward": 0},
            {"goal_reward": np.inf},
            {"step_penalty": -0.1},
            {"step_penalty": np.nan},
            {"max_episode_steps": 0},
            {"max_episode_steps": 1.5},
            {"max_episode_steps": True},
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                with self.assertRaises((ValueError, TypeError)):
                    RoomEnv(**arguments)

    def test_invalid_reset_positions_and_options(self):
        env = RoomEnv(grid_shape=(3, 3), step_size=0.5)
        invalid = (
            {"unknown": [0, 0]},
            {"start": [0, 0]},
            {"goal": [2, 2]},
            {"start_cell": [0]},
            {"goal_cell": [0, 0, 0]},
            {"start_cell": [-1, 0]},
            {"goal_cell": [3, 0]},
            {"start_cell": [0.5, 0]},
            {"goal_cell": [1.0, 0.0]},
            {"goal_cell": [np.nan, 0]},
            {"start_cell": [0, 0], "goal_cell": [0, 0]},
        )
        for options in invalid:
            with self.subTest(options=options):
                with self.assertRaises((ValueError, TypeError)):
                    env.reset(options=options)

    def test_invalid_actions(self):
        env = RoomEnv(grid_shape=(3, 3))
        env.reset(options={"start_cell": [0, 0], "goal_cell": [2, 2]})
        for action in (-1, 4, 0.5, "0"):
            with self.subTest(action=action):
                with self.assertRaises((ValueError, TypeError)):
                    env.step(action)


if __name__ == "__main__":
    unittest.main()
