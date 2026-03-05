"""Code for the SceneTokens loss function."""

import torch
from omegaconf import DictConfig

from scenetokens.models.criterion import Criterion, Reconstruction, TrajectoryPrediction
from scenetokens.schemas.output_schemas import ModelOutput


class SceneTokens(Criterion):
    """Criterion for SceneTokens."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

        self.reconstruction_criterion = Reconstruction(config)
        self.trajpred_criterion = TrajectoryPrediction(config)

    def forward(self, model_output: ModelOutput) -> torch.Tensor:
        """Compute the SceneTokens loss.

        Notation:
            L_rec: scenario-tokenization reconstruction loss
            L_traj: trajectory prediction loss

        Args:
            model_output (ModelOutput): Structured model outputs.

        Returns:
            torch.Tensor: Scalar loss value ``L_rec + L_traj``.
        """
        reconstruction_loss = self.reconstruction_criterion(model_output)
        trajpred_loss = self.trajpred_criterion(model_output)
        return (reconstruction_loss + trajpred_loss).mean()
