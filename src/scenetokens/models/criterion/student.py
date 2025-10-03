import torch
from omegaconf import DictConfig

from scenetokens.models.criterion import Criterion, Reconstruction, TrajectoryPrediction
from scenetokens.schemas.output_schemas import ModelOutput


class Student(Criterion):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

        self.reconstruction_criterion = Reconstruction(config)
        self.trajpred_criterion = TrajectoryPrediction(config)

    def forward(self, outputs: ModelOutput) -> torch.Tensor:
        """Computes the Quantized Student loss which combines the quantizatio loss from the scenario classifier and the
        trajectory predicton loss from the scenario decder.

        Args:
            outputs (ModelOutput): pydantic validator for model outputs.

        Returns:
            loss (torch.tensor): loss value.
        """
        reconstruction_loss = self.reconstruction_criterion(outputs)
        trajpred_loss = self.trajpred_criterion(outputs)
        return (reconstruction_loss + trajpred_loss).mean()
