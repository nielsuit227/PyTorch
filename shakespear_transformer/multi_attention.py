import torch
from self_attention import Head
from torch import nn


class MultiHead(nn.Module):
    def __init__(
        self,
        n_embeddings: int,
        block_size: int,
        n_heads: int,
        head_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embeddings, block_size, head_size, dropout) for _ in range(n_heads)]
        )
        self.projection = nn.Linear(n_embeddings, n_embeddings)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data = torch.cat([h(data) for h in self.heads], dim=-1)
        data = self.projection(data)
        return self.dropout(data)
