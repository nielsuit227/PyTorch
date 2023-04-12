import torch
from stack import NBeatsStack
from torch import nn


class NBeats(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_stacks: int = 20,
        num_blocks: int = 20,
        num_layers: int = 4,
        hidden_units: int = 512,
    ) -> None:
        """
        This implements the full NBeats model [1]. The default parameters are taken
        from the original paper.
        As noted in the paper, there are two configurations of this model.
        1. Generic deep learning
        > Here, there are no restrictions on the basis vectors of the block. This is
        > currently implemented
        2. Interpretable inductive bias
        > This splits the stacks in two, one to cover the trend, one to cover the seasonality.
        > The trend stacks have a basis layer of T = [1, t, ..., t ** p] where p is a parameter
        > and the forward is the dot product between T and theta.

        Parameters
        ----------
        input_dim : int
            Input dimension
        output_dim : int
            Output dimension
        num_stacks : int, default = 20
            Number of stacks
        num_blocks : int, default = 20
            Number of blocks per stack
        num_layers : int, default = 4
            Number of fully connected layers in a block before splitting into backwards and forwards
        hidden_units : int, default = 512
            Number of hidden_units in the fully connected block layers

        [1]: https://arxiv.org/pdf/1905.10437.pdf
        """
        super().__init__()

        if num_stacks < 1:
            raise ValueError("Number of stacks cannot be smaller than 1.")

        self.stacks = [
            NBeatsStack(input_dim, output_dim, hidden_units, num_blocks)
        ] * num_stacks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor | None = None
        forward: torch.Tensor

        for stack in self.stacks:
            x, forward = stack(x)

            if not output:
                output = forward
            else:
                output += forward

        return forward
