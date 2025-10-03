"""Tokenization Class via Vector Quantization.
Vector Quantization paper: https://arxiv.org/abs/1711.00937
"""

import torch
from torch import nn
from vector_quantize_pytorch import VectorQuantize

from scenetokens.schemas.output_schemas import TokenizationOutput


class QuantizedTokenizer(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        num_queries: int,
        hidden_size: int,
        num_tokens: int,
        quantization_weight: float = 0.25,
        commitment_weight: float = 0.25,
        reduction_factor: int = 2,
        *,
        normalize: bool = False,
    ) -> None:
        """Initializes the Naive Tokenizer.

        Args:
            num_queries (int): number of input (learnable) tokenizer queries.
            hidden_size (int): hidden size of the learnable queries.
            num_tokens (int): number of tokens to be learned.
            quantization_weight (float): pulls the quantized values toward the pre-quantized values.
            commitment_weight (float): prevents the pre-quantized values from being too far from the quantized values.
            reduction_factor (int): factor by which to compress the input.
            normalize (bool): normalize the module's input and output.
        """
        super().__init__()
        self.num_tokens = num_tokens
        self.num_queries = num_queries

        # Loss weights
        self.quantization_weight = quantization_weight
        self.commitment_weight = commitment_weight

        # Tokenizer is a VQ-VAE
        reduced_hidden_size = hidden_size // reduction_factor

        # Normalizes and encodes the input value
        if normalize:
            self.prepare_input = nn.Tanh()
            self.encoder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, reduced_hidden_size)
            )

            # Decodes the input value
            self.decoder = nn.Sequential(
                nn.Linear(reduced_hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.Tanh()
            )
        else:
            self.prepare_input = nn.Identity()
            self.encoder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, reduced_hidden_size)
            )

            # Decodes the input value
            self.decoder = nn.Sequential(
                nn.Linear(reduced_hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )

        # Quantization codebook
        self.quantize = VectorQuantize(
            dim=reduced_hidden_size,
            codebook_size=num_tokens,
            commitment_weight=self.commitment_weight,
        )

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
                quantized_embedding (torch.tensor(B, Q, H) | None): discretized embedding values.
        """
        # input shape: (B, Q, H)
        batch_size, num_queries, _ = x.shape

        # Prepares the input for quantization. It is either a Tanh() to map it from [-1, 1] or an Identity layer.
        x = self.prepare_input(x)

        # Encode the embedding.
        # encoder output shape (B, Q, C).
        embedding_prequant = self.encoder(x)

        # Quantize the embedding
        # quantized output shape: (B, Q, C)
        # indices shape: (B, Q)
        # quantization loss shape: (1,)
        quantized_embedding, token_indices, vq_loss = self.quantize(embedding_prequant)

        # Decode the quantized embedding
        # decoder output shape: (B, Q, H)
        embedding_postquant = self.decoder(quantized_embedding)
        embedding_postquant = embedding_postquant.view(batch_size, num_queries, -1)

        return TokenizationOutput(
            num_tokens=self.num_tokens,
            token_indices=token_indices,
            input_embedding=x,
            reconstructed_embedding=embedding_postquant,
            quantized_embedding=quantized_embedding,
            loss=vq_loss,
        )
