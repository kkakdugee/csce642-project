# csce-642-final-project

This project studies how a recurrent network tracks position from movement in
2D and 3D navigation tasks.

## Setup

With Python 3.11 or newer, run `python -m venv .venv`. Activate it with
`.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate`
on macOS or Linux. Then run `python -m pip install -e .`.

## Usage

```python
from navigation.envs.room import RoomEnv

env = RoomEnv(grid_shape=(11, 11))
env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(0)  # Move along +x.
env.close()
```

The [position network](navigation/position_network.py) takes a starting cue
and a sequence of displacements and predicts place-cell activity at each step.

## Position-network pretraining

```sh
python -m navigation.pretrain --grid-shape 5 5 --output runs/position-2d.pt
python -m navigation.pretrain --grid-shape 5 5 5 --output runs/position-3d.pt
```

Each CPU run trains on the starting cue and subsequent moves. It reports
held-out cross-entropy and mean distance to the true position after each move.
The decoder averages the place-cell centers weighted by predicted probabilities.
Use `--help` for settings. `navigation.pretrain.load_checkpoint(path)` returns
the trained model, place-cell population, and configuration.
