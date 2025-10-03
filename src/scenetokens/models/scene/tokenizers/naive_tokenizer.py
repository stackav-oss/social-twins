import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import nn

from scenetokens.schemas.output_schemas import TokenizationOutput


class NaiveTokenizer(nn.Module):
    def __init__(self, num_queries: int, hidden_size: int, num_tokens: int) -> None:
        """Initializes the Naive Tokenizer.

        Args:
            num_queries (int): number of input (learnable) tokenizer queries.
            hidden_size (int): hidden size of the learnable queries.
            num_tokens (int): number of tokens to be learned.
        """
        super().__init__()

        # Prototype AE classifier
        self.encoder = nn.Linear(num_queries * hidden_size, num_tokens)
        self.selu = nn.SELU(inplace=True)
        self.decoder = nn.Linear(num_tokens, num_queries * hidden_size)

    def forward(self, x: torch.Tensor) -> TokenizationOutput:
        """Tokenizes the scenario information.
           B: batch size
           Q: number of queries
           H: hidden size
           C: number of tokens/classes

        Args:
            x (torch.tensor(B, Q, H)): tensor containing the scene encoded information.

        Returns:
            TokenizationOutput: pydantic validator for the tokenizer with:
                token_probabilities (torch.tensor(B, Q, C)): proability mass over the set of tokens/classes.
                token_indices (torch.tensor(B, 1)): selected token/class.
                input_embedding (torch.tensor(B, Q, H) | None): embedding values before encoding.
                reconstructed_embedding (torch.tensor(B, Q, H) | None): embedding values after decoding.
        """
        # input shape: (B, Q, H)
        x_pre = x
        batch_size, num_queries, _ = x.shape
        x = rearrange(x, "b q h -> b (q h)")
        # encoder output shape: (B, C)
        token_raw_embedding_prequant = self.encoder(x)
        self.selu(token_raw_embedding_prequant)
        # token probabilities shape: (B, C)
        token_probabilities = F.softmax(token_raw_embedding_prequant, dim=-1)
        # token_indices shape: (B,)
        token_indices = torch.argmax(token_probabilities, dim=-1)
        # decoder output shape: (B, Q * C) -> (B, Q, C)
        token_raw_embedding_postquant = self.decoder(token_raw_embedding_prequant)
        token_raw_embedding_postquant = token_raw_embedding_postquant.view(batch_size, num_queries, -1)
        return TokenizationOutput(
            token_probabilities=token_probabilities,
            token_indices=token_indices,
            input_embedding=x_pre,
            reconstructed_embedding=token_raw_embedding_postquant,
        )
