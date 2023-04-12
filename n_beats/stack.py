import torch
from torch import nn

from .block import NBeatsBlock


class NBeatsStack(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, hidden_units: int, num_blocks: int
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("Number of blocks cannot be smaller than 1.")
        self.stack = [
            NBeatsBlock(input_dim, output_dim, hidden_units) for i in range(num_blocks)
        ]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Initialize output tensor
        output: torch.Tensor | None = None

        # Iterate through blocks
        for block in self.stack:

            # Get the block outputs
            back, forward = block(x)

            # Subtract back from input / residual
            x -= back

            # Update output
            if not output:
                output = forward
            else:
                output += forward

        if not output:
            raise ValueError("Empty stack, output is not calculated")

        return x, output
