"""Fixed-length pretraining walks using the navigation room's movement rules."""

import operator

import numpy as np
from numpy.typing import NDArray

from navigation.envs.movement import move_cell
from navigation.place_cells import PlaceCells


def generate_batch(
    *,
    grid_shape: tuple[int, ...],
    step_size: float,
    place_cells: PlaceCells,
    batch_size: int,
    sequence_length: int,
    rng: np.random.Generator,
    include_positions: bool = False,
) -> dict[str, NDArray[np.floating]]:
    """Return initial_code (B,N), displacements (B,T,D) and targets (B,T,N).

    Starts and attempted axial actions are uniform. Rejected wall moves remain
    zero-displacement steps. Target t encodes the position after displacement t.
    Walks have no goals, rewards or early termination. Use separate generators
    for training and validation while sharing the same place-cell population.
    For validation, include_positions adds the true post-move coordinates (B,T,D).
    All sizes, physical units and sampling generators are explicit; this
    function does not select an experimental protocol.
    """
    grid_shape = tuple(operator.index(n) for n in grid_shape)
    dimensions = len(grid_shape)
    if dimensions not in (2, 3) or min(grid_shape) < 2:
        raise ValueError(
            "grid_shape must have two or three integer axes of at least two."
        )

    step_size = float(step_size)
    largest_float32 = np.finfo(np.float32).max
    if not 0 < step_size <= largest_float32 or np.float32(step_size) == 0:
        raise ValueError(
            "step_size must be finite, positive and representable in float32."
        )

    bounds = (np.asarray(grid_shape) - 1) * step_size
    if not np.all(np.isfinite(bounds)):
        raise ValueError("Room bounds must be finite.")
    if place_cells.centers.shape[1] != dimensions:
        raise ValueError("Place-cell and room dimensions must match.")

    counts = (batch_size, sequence_length)
    if any(isinstance(count, (bool, np.bool_)) for count in counts):
        raise ValueError("batch_size and sequence_length must be positive integers.")
    batch_size = operator.index(batch_size)
    sequence_length = operator.index(sequence_length)
    if batch_size < 1 or sequence_length < 1:
        raise ValueError("batch_size and sequence_length must be positive integers.")

    trajectory_shape = (batch_size, sequence_length + 1, dimensions)
    grid_cells = np.empty(trajectory_shape, dtype=np.int64)
    grid_cells[:, 0] = rng.integers(grid_shape, size=(batch_size, dimensions))
    actions = rng.integers(2 * dimensions, size=(batch_size, sequence_length))

    for batch_index in range(batch_size):
        for step_index in range(sequence_length):
            current_cell = grid_cells[batch_index, step_index]
            action = actions[batch_index, step_index]
            next_cell = move_cell(current_cell, action, grid_shape)
            grid_cells[batch_index, step_index + 1] = next_cell

    positions = grid_cells * step_size
    codes = place_cells.encode(positions)
    cell_displacements = np.diff(grid_cells, axis=1)
    displacements = cell_displacements * step_size
    displacements = displacements.astype(np.float32)

    batch = {
        "initial_code": codes[:, 0],
        "displacements": displacements,
        "targets": codes[:, 1:],
    }
    if include_positions:
        batch["positions"] = positions[:, 1:]
    return batch
