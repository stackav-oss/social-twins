"""Criterion for the encoder-only model. Uses only VQ-VAE reconstruction loss."""

import torch
from omegaconf import DictConfig

from scenetokens.models.criterion.base_criterion import Criterion
from scenetokens.models.criterion.reconstruction import Reconstruction
from scenetokens.schemas.output_schemas import ModelOutput


class EncoderOnly(Criterion):
    """Encoder-only criterion using VQ-VAE reconstruction loss."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)
        self.reconstruction_criterion = Reconstruction(config)

    def forward(self, model_output: ModelOutput) -> torch.Tensor:
        """Computes the clustering loss which is purely the VQ-VAE reconstruction + quantization loss.

        No trajectory prediction loss is included - the model learns to cluster scenarios based on
        the reconstruction objective alone.

        Args:
            model_output (ModelOutput): pydantic validator for model outputs.

        Returns:
            loss (torch.tensor): loss value.
        """
        reconstruction_loss = self.reconstruction_criterion(model_output)
        return reconstruction_loss.mean()
