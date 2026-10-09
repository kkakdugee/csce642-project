# csce-642-final-project

Research code for studying learned spatial representations in 2D and 3D navigation.

## Installation

Requires Python 3.11 or newer. From the repository root, create a virtual
environment:

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` on macOS/Linux, then install:

```sh
python -m pip install -e .
```

## Usage

```python
from navigation.envs.room import RoomEnv

env = RoomEnv(grid_shape=(11, 11), step_size=1.0)
obs, info = env.reset(seed=0, options={"start_cell": (0, 0), "goal_cell": (1, 0)})
obs, reward, terminated, truncated, info = env.step(0)  # +x reaches the goal
assert terminated and info["is_success"]
env.close()
```
