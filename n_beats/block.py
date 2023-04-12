import torch
from torch import nn


class NBeatsBlock(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, hidden_units: int, num_layers: int
    ) -> None:
        """
        Parameters
        ----------
        input_dim : int
            The input dimension
        output_dim : int
            The output dimension
        hidden_units : int
            The number of hidden units.
        num_layers : int
            The number of fully connected layers (including first from input > width)
        """
        super().__init__()

        if num_layers < 1:
            raise ValueError("Number of layers cannot be smaller than 1.")
        if hidden_units < 1:
            raise ValueError("Number of hidden units cannot be smaller than 1.")

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            *[nn.Linear(hidden_units, hidden_units), nn.ReLU()] * (num_layers - 1)
        )
        self.project_backward = nn.Linear(hidden_units, hidden_units, bias=False)
        self.project_forward = nn.Linear(hidden_units, hidden_units, bias=False)
        self.basis_vector_backward = nn.Linear(hidden_units, input_dim, bias=False)
        self.basis_vector_forward = nn.Linear(hidden_units, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.layers(x)

        residual = self.project_backward(x)
        residual = self.basis_vector_backward(residual)

        forward = self.project_forward(x)
        forward = self.basis_vector_forward(forward)

        # NOTE: we directly subtract the input,
        return residual - x, forward
