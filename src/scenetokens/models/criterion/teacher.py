import torch
from omegaconf import DictConfig

from scenetokens.models.criterion import (
    Classification,
    Criterion,
    FocalClassification,
    Reconstruction,
    TrajectoryPrediction,
)
from scenetokens.schemas.output_schemas import ModelOutput


class Teacher(Criterion):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

        self.reconstruction_criterion = Reconstruction(config)
        self.trajpred_criterion = TrajectoryPrediction(config)

        # Classification loss is used for causal agent prediction. If 'use_focal_loss' will use FocalClassification
        # which focuses on imbalanced classes
        self.use_focal_loss = config.get("use_focal_loss", False)
        self.classification_loss = FocalClassification(config) if self.use_focal_loss else Classification(config)

    def forward(self, outputs: ModelOutput) -> torch.Tensor:
        """Computes the Quantized Teacher loss which combines the quantizatio loss from scenario and agent tokenization,
        the trajectory prediction loss and the mask classifier loss.

        Args:
            outputs (ModelOutput): pydantic validator for model outputs.

        Returns:
            loss (torch.tensor): loss value.
        """
        reconstruction_loss = self.reconstruction_criterion(outputs)
        trajpred_loss = self.trajpred_criterion(outputs)
        classification_loss = self.classification_loss(outputs)
        return (reconstruction_loss + trajpred_loss + classification_loss).mean()
