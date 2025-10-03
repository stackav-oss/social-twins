from abc import ABC, abstractmethod

from omegaconf import DictConfig
from torch import Tensor, nn

from scenetokens.schemas.output_schemas import ModelOutput


class Criterion(nn.Module, ABC):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, inputs: ModelOutput) -> Tensor:
        """Computes the loss function."""
