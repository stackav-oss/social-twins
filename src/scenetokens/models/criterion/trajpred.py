import torch
import torch.nn.functional as F  # noqa: N812
from omegaconf import DictConfig

from scenetokens.models.criterion.base_criterion import Criterion
from scenetokens.schemas.output_schemas import ModelOutput


class TrajectoryPrediction(Criterion):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

        self.rho_limit = config.get("rho_limit", 0.5)
        self.log_std = config.get("log_std", -1.609)
        self.log_range = config.get("log_range", 5.0)
        self.use_square_gmm = config.get("use_square_gmm", False)
        self.pre_nearest_mode_idxs = config.get("pre_nearest_mode_idxs", None)
        self.timestamp_loss_weight = config.get("timestamp_loss_weight", None)
        self.trajpred_weight = config.get("trajpred_weight", 1.0)

        self.square_gmm_size = 3
        self.nonsquare_gmm_size = 5  # (µ_x, µ_y, sig_x, sig_y, p)

    def forward(self, model_output: ModelOutput) -> torch.Tensor:
        """GMM Loss
            B: batch_size, M: number of modes, F: number of future steps, D = 5: gmm dimensions

        Args:
            model_output (ModelOutput): dictionary containing tensor elements needed to compute the loss function.
                - predicted_trajectory (torch.tensor(B, M, F, D=5)): decoded trajectories, D=5 is the distribution.
                - predicted_probability (torch.tensor(B, M)): mode's predicted probability.
                - gt_trajectory (torch.tensor(B, F, D=2)): ground truth trajectories, where D=2 is (x, y)

        Returns:
            loss (torch.tensor): loss value.
        """
        # Predicted scores for each trajectory shape: (B, num_modes)
        trajectory_decoder_output = model_output.trajectory_decoder_output
        pred_scores = trajectory_decoder_output.mode_logits.value
        # Predicted trajectories shape: (B, num_modes, future_len, 5)
        pred_trajs = trajectory_decoder_output.decoded_trajectories.value
        # Ground truth trajectories shape: (B, num_modes)
        gt_trajs = model_output.future_ground_truth.value

        if self.use_square_gmm:
            assert pred_trajs.shape[-1] == self.square_gmm_size
        else:
            assert pred_trajs.shape[-1] == self.nonsquare_gmm_size

        batch_size = pred_trajs.shape[0]
        # Ground truth valid flags shape: (B, F)
        gt_valid_mask = gt_trajs[..., -1]

        if self.pre_nearest_mode_idxs is not None:
            nearest_mode_idxs = self.pre_nearest_mode_idxs
        else:
            # Ground truth valid flags shape: (B, M, F, D=2) (x, y)
            distance = (pred_trajs[:, :, :, 0:2] - gt_trajs[:, None, :, :2]).norm(dim=-1)
            # Ground truth valid flags shape: (B, M, F)
            distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1)
            nearest_mode_idxs = distance.argmin(dim=-1)

        # Nearest mode shape: (B)
        nearest_mode_bs_idxs = torch.arange(batch_size).type_as(nearest_mode_idxs)

        # Nearest trajectories shape: (B, F, 5)
        nearest_trajs = pred_trajs[nearest_mode_bs_idxs, nearest_mode_idxs]
        # Residual trajectory shape: (B, F, 2)
        res_trajs = gt_trajs[..., :2] - nearest_trajs[:, :, 0:2]
        # dx, dy trajectory shapes: (B, F)
        dx = res_trajs[:, :, 0]
        dy = res_trajs[:, :, 1]

        if self.use_square_gmm:
            # All stats values shapes: (B, F)
            log_std1 = log_std2 = torch.clip(nearest_trajs[:, :, 2], min=self.log_std, max=self.log_range)
            std1 = std2 = torch.exp(log_std1)  # (0.2m to 150m)
            rho = torch.zeros_like(log_std1)
        else:
            log_std1 = torch.clip(nearest_trajs[:, :, 2], min=self.log_std, max=self.log_range)
            log_std2 = torch.clip(nearest_trajs[:, :, 3], min=self.log_std, max=self.log_range)
            std1 = torch.exp(log_std1)  # (0.2m to 150m)
            std2 = torch.exp(log_std2)  # (0.2m to 150m)
            rho = torch.clip(nearest_trajs[:, :, 4], min=-self.rho_limit, max=self.rho_limit)

        gt_valid_mask = gt_valid_mask.type_as(pred_scores)
        if self.timestamp_loss_weight is not None:
            gt_valid_mask = gt_valid_mask * self.timestamp_loss_weight[None, :]

        # Regression loss: (B, F)
        # -log(a^-1 * e^b) = log(a) - b
        reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho**2)  # (batch_size, num_timestamps)
        reg_gmm_exp = (0.5 * 1 / (1 - rho**2)) * (
            (dx**2) / (std1**2) + (dy**2) / (std2**2) - 2 * rho * dx * dy / (std1 * std2)
        )  # (batch_size, num_timestamps)

        # Regression and classification losses shape: (B)
        reg_loss = ((reg_gmm_log_coefficient + reg_gmm_exp) * gt_valid_mask).sum(dim=-1)
        loss_cls = F.cross_entropy(input=pred_scores, target=nearest_mode_idxs, reduction="none")

        # Final los shape: (1)
        return self.trajpred_weight * (reg_loss + loss_cls).mean()
