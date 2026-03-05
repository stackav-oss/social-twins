"""Code for the CausalSceneTokens loss function."""

import torch
from omegaconf import DictConfig

from scenetokens.models.criterion import (
    CausalClassification,
    Criterion,
    FocalCausalClassification,
    Reconstruction,
    TrajectoryPrediction,
)
from scenetokens.schemas.output_schemas import ModelOutput


class CausalSceneTokens(Criterion):
    """Criterion for CausalSceneTokens.

    The total loss is the sum of reconstruction, trajectory prediction, and
    causal classification losses.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

        self.reconstruction_criterion = Reconstruction(config)
        self.trajpred_criterion = TrajectoryPrediction(config)

        # Classification loss for causal-agent labels.
        # If ``use_focal_loss`` is enabled, use focal loss for class-imbalanced data.
        self.use_focal_loss = config.get("use_focal_loss", False)
        self.classification_loss = (
            FocalCausalClassification(config) if self.use_focal_loss else CausalClassification(config)
        )

    def forward(self, model_output: ModelOutput) -> torch.Tensor:
        """Compute the CausalSceneTokens loss.

        Notation:
            L_rec: reconstruction loss
            L_traj: trajectory prediction loss
            L_cls: causal classification loss

        Args:
            model_output (ModelOutput): Structured model outputs.

        Returns:
            torch.Tensor: Scalar loss value ``L_rec + L_traj + L_cls``.
        """
        reconstruction_loss = self.reconstruction_criterion(model_output)
        trajpred_loss = self.trajpred_criterion(model_output)
        classification_loss = self.classification_loss(model_output)
        return (reconstruction_loss + trajpred_loss + classification_loss).mean()
