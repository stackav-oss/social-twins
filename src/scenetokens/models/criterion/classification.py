import torch
from omegaconf import DictConfig, ListConfig

from scenetokens.models.criterion.base_criterion import Criterion
from scenetokens.schemas.output_schemas import ModelOutput


class Classification(Criterion):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

        self.classification_weight = config.get("classification_weight", 1.0)

        self.loss_function = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, model_output: ModelOutput) -> torch.Tensor:
        """Computes the Binary Cross Entropy loss for the causal values.
           B: batch size
           C: number of classes

        Args:
            model_output (ModelOutput): pydantic validator for model outputs.

        Returns:
            loss (torch.tensor): loss value.
        """
        causal_output = model_output.causal_output

        # Target has shape (B, N)
        gt = causal_output.causal_gt.value.view(-1).long()
        # Logits has shape (B, N, C)
        logits = causal_output.causal_logits.value.view(-1, 2)
        return (self.classification_weight * self.loss_function(logits, gt)).mean()


class FocalClassification(Criterion):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

        self.classification_weight = config.get("classification_weight", 1.0)
        self.gamma = config.get("gamma", 2.0)

        self.num_classes = config.get("num_classes", 2)

        self.alpha = config.get("alpha", [0.25, 1.0])
        assert self.alpha is not None, "Alpha value(s) must be provided as a single or per-class value."
        if isinstance(self.alpha, (list, tuple, ListConfig)):
            assert len(self.alpha) == self.num_classes, (
                f"Need a per-class alpha value. Num classes {self.num_classes}, num alphas: {len(self.alpha)}"
            )
            self.alpha = torch.tensor(self.alpha, dtype=torch.float32)
        else:
            # if a single number, propagate per class
            self.alpha = torch.tensor([self.alpha] * self.num_classes, dtype=torch.float32)
        self.loss_function = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, model_output: ModelOutput) -> torch.Tensor:
        """Computes the focal loss which is the binary cross entropy loss for imbalanced classes. It focuses on
            missclassified elements. Reference: https://arxiv.org/pdf/1708.02002
            B: batch size
            C: number of tokens/classes

        Args:
            model_output (ModelOutput): pydantic validator for model outputs.

        Returns:
            loss (torch.tensor): loss value.
        """
        causal_output = model_output.causal_output

        # Target has shape (B, N)
        gt = causal_output.causal_gt.value.view(-1).long()

        # Logits has shape (B, N, C)
        logits = causal_output.causal_logits.value.view(-1, 2)

        # Apply the cross entropy loss
        ce_loss = self.loss_function(logits, gt)
        pt = torch.exp(-ce_loss)

        # Apply alpha weighing
        self.alpha = self.alpha.to(logits.device)

        # Get alpha value per sample
        alpha = self.alpha[gt]
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        return self.classification_weight * focal_loss.mean()
