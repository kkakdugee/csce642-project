"""Fixed Gaussian place-cell populations in physical 2D or 3D coordinates.

Activity is softmax(-squared_distance / (2 * sigma**2)) over cells, matching
the softmax encoding in google-deepmind/grid-cells/ensembles.py.
"""

import operator

import numpy as np
from numpy.typing import ArrayLike, NDArray


class PlaceCells:
    """Encode positions using fixed centers and a width in physical units."""

    def __init__(self, centers: ArrayLike, sigma: float):
        self.centers = np.array(centers, dtype=np.float64, copy=True)
        if self.centers.ndim != 2:
            raise ValueError("centers must have shape (n_cells, 2) or (n_cells, 3).")

        n_cells, dimensions = self.centers.shape
        if dimensions not in (2, 3):
            raise ValueError("centers must have shape (n_cells, 2) or (n_cells, 3).")
        if n_cells < 1 or not np.all(np.isfinite(self.centers)):
            raise ValueError("centers must contain at least one cell and be finite.")

        self.sigma = float(sigma)
        if not 0 < self.sigma < np.inf:
            raise ValueError("sigma must be finite and positive.")
        self.centers.setflags(write=False)

    @classmethod
    def sample(
        cls,
        *,
        n_cells: int,
        bounds: ArrayLike,
        sigma: float,
        rng: np.random.Generator,
    ) -> "PlaceCells":
        """Sample centers uniformly between zero and each physical upper bound.

        Pass a generator separate from those used for trajectory sampling.
        Reuse the resulting population for cues, targets and all data splits.
        """
        if isinstance(n_cells, (bool, np.bool_)):
            raise ValueError("n_cells must be a positive integer.")
        n_cells = operator.index(n_cells)
        if n_cells < 1:
            raise ValueError("n_cells must be a positive integer.")

        bounds = np.asarray(bounds, dtype=np.float64)
        if bounds.shape not in ((2,), (3,)):
            raise ValueError(
                "bounds must contain two or three finite positive lengths."
            )
        valid_bounds = (bounds > 0) & np.isfinite(bounds)
        if not np.all(valid_bounds):
            raise ValueError(
                "bounds must contain two or three finite positive lengths."
            )

        centers = rng.uniform(0, bounds, size=(n_cells, bounds.size))
        return cls(centers, sigma)

    def encode(self, positions: ArrayLike) -> NDArray[np.float32]:
        """Map positions (..., dimensions) to normalized activity (..., n_cells)."""
        positions = np.asarray(positions, dtype=np.float64)
        dimensions = self.centers.shape[1]
        if positions.ndim < 1 or positions.shape[-1] != dimensions:
            raise ValueError(
                "positions must have one coordinate per spatial dimension."
            )
        if not np.all(np.isfinite(positions)):
            raise ValueError("positions must be finite.")

        # Add a cell axis to compare each position with every center.
        with np.errstate(over="raise", invalid="raise"):
            offsets = positions[..., None, :] - self.centers
            squared_distances = np.sum(offsets**2, axis=-1)

        # Remove the shared offset before scaling, so narrow fields retain a
        # finite maximum even when all unshifted Gaussian responses underflow.
        nearest_distance = squared_distances.min(axis=-1, keepdims=True)
        squared_distances -= nearest_distance

        with np.errstate(over="ignore", under="ignore"):
            # Divide twice: squaring an extreme sigma could underflow or overflow.
            log_activity = -0.5 * squared_distances / self.sigma / self.sigma
            activity = np.exp(log_activity)

        total_activity = activity.sum(axis=-1, keepdims=True)
        activity /= total_activity
        return activity.astype(np.float32)
