"""A bounded 2D/3D room with fixed-distance axial movement.

This is the simulator, not the complete agent observation wrapper. Observations
contain displacement and previous action only. Coordinates in ``info`` are
privileged: they support future cue encoding, supervised targets and evaluation.
"""

import operator

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.error import ResetNeeded

from navigation.envs.movement import move_cell


class RoomEnv(gym.Env):
    """Navigate on a spatial grid with whole-move boundary rejection.

    ``grid_shape`` counts positions per axis, including both boundaries.
    Coordinates run from zero to ``(grid_shape - 1) * step_size``. Defaults
    are development settings, not an experimental protocol.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        grid_shape: tuple[int, ...] = (11, 11),
        step_size: float = 1.0,
        goal_radius: float = 0.25,
        max_episode_steps: int = 100,
        goal_reward: float = 1.0,
        step_penalty: float = 0.01,
    ):
        super().__init__()
        self.grid_shape = tuple(operator.index(n) for n in grid_shape)
        if len(self.grid_shape) not in (2, 3):
            raise ValueError("grid_shape must have two or three axes.")
        if min(self.grid_shape) < 2:
            raise ValueError("Each grid_shape entry must be an integer of at least two.")
        self.step_size = float(step_size)
        displacement_space = spaces.Box(
            -self.step_size, self.step_size, shape=(len(self.grid_shape),), dtype=np.float32
        )
        if not 0 < displacement_space.high[0] < np.inf:
            raise ValueError("step_size must be finite, positive and representable in float32.")
        bounds = (np.asarray(self.grid_shape) - 1) * self.step_size
        if not np.all(np.isfinite(bounds)):
            raise ValueError("Room bounds must be finite.")
        if not 0 < goal_radius < np.linalg.norm(bounds):
            raise ValueError("goal_radius must be positive and smaller than the room diagonal.")
        self.max_episode_steps = operator.index(max_episode_steps)
        if isinstance(max_episode_steps, bool) or self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be a positive integer.")
        if not 0 < goal_reward < np.inf:
            raise ValueError("goal_reward must be finite and positive.")
        if not 0 <= step_penalty < np.inf:
            raise ValueError("step_penalty must be finite and nonnegative.")
        self.goal_radius = float(goal_radius)
        self.goal_reward = float(goal_reward)
        self.step_penalty = float(step_penalty)
        self.action_space = spaces.Discrete(2 * len(self.grid_shape))
        self.observation_space = spaces.Dict(
            {
                "displacement": displacement_space,
                "previous_action": spaces.Discrete(self.action_space.n + 1),
            }
        )
        self._needs_reset = True

    def reset(self, *, seed=None, options=None):
        """Start navigation; optional start_cell/goal_cell are integer grid indices.

        Reset is the cue/setup boundary. It consumes no movement, reward or
        navigation steps. An agent wrapper will encode and store the goal first,
        then initialize its tracker from the returned starting position.
        """
        super().reset(seed=seed)
        options = {} if options is None else options
        if options.keys() - {"start_cell", "goal_cell"}:
            raise ValueError("Supported reset options are 'start_cell' and 'goal_cell'.")
        start = self._parse_cell(options.get("start_cell"))
        goal = self._parse_cell(options.get("goal_cell"))
        self._cell, self._goal_cell = self._sample_task(start, goal)
        self._elapsed_steps = 0
        self._needs_reset = False
        observation = {
            "displacement": np.zeros(len(self.grid_shape), dtype=np.float32),
            "previous_action": int(self.action_space.n),
        }
        return observation, self._info(False)

    def step(self, action):
        if self._needs_reset:
            raise ResetNeeded("Call reset() before stepping or after an episode finishes.")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action!r}")
        action = int(action)
        next_cell = move_cell(self._cell, action, self.grid_shape)
        displacement = ((next_cell - self._cell) * self.step_size).astype(np.float32)
        self._cell = next_cell
        self._elapsed_steps += 1
        terminated = self._at_goal(self._cell, self._goal_cell)
        truncated = self._elapsed_steps >= self.max_episode_steps and not terminated
        self._needs_reset = terminated or truncated
        reward = -self.step_penalty + (self.goal_reward if terminated else 0.0)
        observation = {"displacement": displacement, "previous_action": action}
        return observation, reward, terminated, truncated, self._info(terminated)

    def _parse_cell(self, cell):
        if cell is None:
            return None
        cell = np.array([operator.index(n) for n in cell], dtype=np.int64)
        if cell.shape != (len(self.grid_shape),):
            raise ValueError("Cell must have one integer index per axis.")
        if np.any(cell < 0) or np.any(cell >= self.grid_shape):
            raise ValueError("Cell lies outside the room.")
        return cell

    def _at_goal(self, cell, goal):
        """Use the same inclusive radius for task sampling and goal entry."""
        distance = np.linalg.norm((cell - goal) * self.step_size)
        return bool(
            distance <= self.goal_radius
            or np.isclose(distance, self.goal_radius, rtol=1e-12, atol=0)
        )

    def _sample_task(self, start, goal):
        """Sample grid positions, excluding tasks already inside the goal."""
        attempts = 1 if start is not None and goal is not None else 1000
        for _ in range(attempts):
            start_cell = self.np_random.integers(self.grid_shape) if start is None else start
            goal_cell = self.np_random.integers(self.grid_shape) if goal is None else goal
            if not self._at_goal(start_cell, goal_cell):
                return start_cell, goal_cell
        raise ValueError(
            "Could not select start/goal outside the goal radius. "
            "Use a smaller radius or provide separated start and goal positions."
        )

    def _info(self, is_success):
        return {
            "position": self._cell.astype(np.float64) * self.step_size,
            "goal": self._goal_cell.astype(np.float64) * self.step_size,
            "elapsed_steps": self._elapsed_steps,
            "is_success": is_success,
        }
