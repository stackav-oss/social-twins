import torch
from omegaconf import DictConfig, ListConfig

from scenetokens.models.criterion.base_criterion import Criterion
from scenetokens.schemas.output_schemas import ModelOutput


class SafetyClassification(Criterion):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

        self.classification_weight = config.get("classification_weight", 1.0)
        self.safety_type = config.get("safety_type", "individual")  # "individual" or "interaction"

        # CrossEntropyLoss supports multi-class classification
        self.loss_function = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, model_output: ModelOutput) -> torch.Tensor:
        """Computes the multi-class Cross Entropy loss.

        B: batch size
        N: number of elements per batch
        C: number of classes

        Args:
            model_output (ModelOutput): pydantic validator for model outputs.

        Returns:
            torch.Tensor: scalar loss value
        """
        safety_output = model_output.safety_output
        if self.safety_type == "individual":
            gt = safety_output.individual_safety_gt.value
            logits = safety_output.individual_safety_logits.value
        elif self.safety_type == "interaction":
            gt = safety_output.interaction_safety_gt.value
            logits = safety_output.interaction_safety_logits.value
        else:
            error_msg = f"Unknown safety_type: {self.safety_type}. Supported types are 'individual' and 'interaction'."
            raise ValueError(error_msg)

        # Ground truth labels: (B, N) → (B*N,)
        gt = gt.view(-1).long()

        # Logits: (B, N, C) → (B*N, C)
        logits = logits.view(-1, logits.shape[-1])
        loss = self.loss_function(logits, gt)
        return self.classification_weight * loss.mean()


class CausalClassification(Criterion):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

        self.classification_weight = config.get("classification_weight", 1.0)

        # CrossEntropyLoss supports multi-class classification
        self.loss_function = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, model_output: ModelOutput) -> torch.Tensor:
        """Computes the multi-class Cross Entropy loss.

        B: batch size
        N: number of elements per batch
        C: number of classes

        Args:
            model_output (ModelOutput): pydantic validator for model outputs.

        Returns:
            torch.Tensor: scalar loss value
        """
        causal_output = model_output.causal_output

        # Ground truth labels: (B, N) → (B*N,)
        gt = causal_output.causal_gt.value.view(-1).long()

        # Logits: (B, N, C) → (B*N, C)
        logits = causal_output.causal_logits.value.view(-1, causal_output.causal_logits.value.shape[-1])

        loss = self.loss_function(logits, gt)
        return self.classification_weight * loss.mean()


class FocalCausalClassification(Criterion):
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
