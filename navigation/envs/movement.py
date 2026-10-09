"""Integer grid transitions shared by navigation and future pretraining.

Axes are x, y, then z. Actions are +x, -x, +y, -y, +z, -z.
The caller supplies a valid cell, action and grid shape. Conversion to physical
coordinates belongs at the environment boundary: position = cell * step_size.
"""

import numpy as np
from numpy.typing import NDArray


def move_cell(
    cell: NDArray[np.int64], action: int, grid_shape: tuple[int, ...]
) -> NDArray[np.int64]:
    """Move one cell along an axis; reject a move that would leave the room."""
    axis, direction = divmod(action, 2)
    coordinate = cell[axis] + (1 if direction == 0 else -1)
    candidate = cell.copy()
    if 0 <= coordinate < grid_shape[axis]:
        candidate[axis] = coordinate
    return candidate
