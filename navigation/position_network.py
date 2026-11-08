"""Estimate place-cell activity from a starting cue and measured displacements."""

from torch import Tensor, nn


class PositionNetwork(nn.Module):
    """A single-layer RNN. The caller holds the recurrent state for each episode."""

    def __init__(self, *, dimensions: int, n_cells: int, hidden_size: int):
        super().__init__()
        if dimensions not in (2, 3):
            raise ValueError("dimensions must be 2 or 3.")
        if n_cells < 1 or hidden_size < 1:
            raise ValueError("n_cells and hidden_size must be positive.")

        self.cue_encoder = nn.Linear(n_cells, hidden_size)
        self.recurrent = nn.RNN(dimensions, hidden_size, batch_first=True)
        self.readout = nn.Linear(hidden_size, n_cells)

    def initial_state(self, initial_code: Tensor) -> Tensor:
        """Start a new episode from a (batch, cells) cue, before any movement."""
        return self.cue_encoder(initial_code).tanh()

    def predict_place_cells(self, activity: Tensor) -> Tensor:
        """Decode cue or movement activity into probabilities over place cells."""
        logits = self.readout(activity)
        return logits.softmax(dim=-1)

    def step(self, displacement: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        """Advance one (batch, dimensions) move. Return new state and prediction."""
        # RNN takes a time axis on inputs and a layer axis on the initial state.
        sequence = displacement.unsqueeze(1)
        activity, _ = self.recurrent(sequence, state.unsqueeze(0))
        next_state = activity[:, 0]
        return next_state, self.predict_place_cells(next_state)

    def forward(
        self, initial_code: Tensor, displacements: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return activity (B, T, hidden_size) and predictions (B, T, n_cells)."""
        if displacements.ndim != 3:
            raise ValueError("displacements must have shape (batch, time, dimensions).")
        state = self.initial_state(initial_code)
        # Output t describes the position after displacement t, as in generate_batch.
        activity, _ = self.recurrent(displacements, state.unsqueeze(0))
        return activity, self.predict_place_cells(activity)
