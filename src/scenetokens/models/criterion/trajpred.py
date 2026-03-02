import torch
import torch.nn.functional as F  # noqa: N812
from omegaconf import DictConfig
from torch.distributions import Laplace, MultivariateNormal

from scenetokens.models.criterion.base_criterion import Criterion
from scenetokens.schemas.output_schemas import ModelOutput


class TrajectoryPrediction(Criterion):
    """Trajectory prediction criterion using a GMM-based objective."""

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
        """Compute GMM-based trajectory prediction loss.

        Notation:
            B: batch_size
            M: number of modes
            F: number of future steps
            Dp: predicted trajectory distribution dimensions (3 or 5)
            Dg: ground-truth trajectory dimensions (x, y, valid_flag)

        Args:
            model_output (ModelOutput): structured model outputs used to compute the loss.
                - decoded_trajectories (torch.Tensor(B, M, F, Dp)): decoded trajectories.
                - mode_logits (torch.Tensor(B, M)): mode logits for classification.
                - future_ground_truth (torch.Tensor(B, F, Dg)): ground-truth trajectories.

        Returns:
            torch.Tensor: scalar loss value.
        """
        # Predicted scores for each trajectory shape: (B, num_modes)
        trajectory_decoder_output = model_output.trajectory_decoder_output
        pred_scores = trajectory_decoder_output.mode_logits.value
        # Predicted trajectories shape: (B, num_modes, future_len, 5)
        pred_trajs = trajectory_decoder_output.decoded_trajectories.value
        # Ground truth trajectories shape: (B, F, Dg)
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

        # Final loss shape: (1)
        return self.trajpred_weight * (reg_loss + loss_cls).mean()


class TrajectoryPredictionAutoBot(Criterion):
    """TrajectoryPredictionAutoBot criterion class.

    Reference: https://arxiv.org/abs/2104.00563.
    Adapted from the UniTraj framework: https://github.com/vita-epfl/UniTraj/blob/main/unitraj/models/autobot/autobot.py
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize the AutoBot trajectory prediction criterion."""
        super().__init__(config=config)

        # These parameters were directly borrowed from the UniTraj implementation and were not tuned for the current
        # setting. They can be further tuned for better performance.
        self.fdeade_weight = config.get("fdeade_weight", 100.0)
        self.kl_weight = config.get("kl_weight", 20.0)
        self.entropy_weight = config.get("entropy_weight", 40.0)
        self.use_fdeade_aux_loss = config.get("use_fdeade_aux_loss", True)

        self.nonsquare_gmm_size = 5  # (µ_x, µ_y, sig_x, sig_y, p)

    def forward(self, model_output: ModelOutput) -> torch.Tensor:
        """Compute the trajectory prediction loss for the AutoBot model.

        Notation:
            B: batch_size
            M: number of modes
            F: number of future steps
            Dp: gmm dimensions (µ_x, µ_y, sig_x, sig_y, p)
            Dg: trajectory dimensions (x, y, valid_flag)

        Args:
            model_output (ModelOutput): structured model outputs used to compute the loss function.
                - decoded_trajectories (torch.Tensor(B, M, F, Dp)): decoded trajectories.
                - mode_probabilities (torch.Tensor(B, M)): predicted mode probabilities.
                - future_ground_truth (torch.Tensor(B, F, Dg)): ground-truth trajectories.

        Returns:
            torch.Tensor: scalar loss value.
        """
        # Extract the relevant trajectory prediction tensors
        trajectory_decoder_output = model_output.trajectory_decoder_output
        pred_scores = trajectory_decoder_output.mode_probabilities.value  # shape: (B, M)
        pred_trajs = trajectory_decoder_output.decoded_trajectories.value  # shape: (B, M, F, Dp)
        gt_trajs = model_output.future_ground_truth.value  # shape: (B, F, Dg)
        gt_valid_mask = gt_trajs[..., -1]  # shape: (B, F)

        _, num_modes = pred_scores.shape

        log_lik_list = []
        with torch.no_grad():
            for mode in range(num_modes):
                nll = self.nll_pytorch_dist(
                    pred_trajs[:, mode], gt_trajs, gt_valid_mask, return_loss=False
                )  # shape: (B, F)
                log_lik_list.append(-nll.unsqueeze(1))  # Add a new dimension to concatenate later

            # Concatenate the list to form the log_lik tensor
            log_likelihood = torch.cat(log_lik_list, dim=1)  # shape: (B, M, F)
            log_priors = torch.log(pred_scores)  # shape: (B, M)
            log_posterior_unnorm = log_likelihood + log_priors.unsqueeze(2)  # shape: (B, M, F) + (B, M, 1) = (B, M, F)

            # Compute logsumexp for normalization, ensuring no in-place operations
            log_sumexp = torch.logsumexp(log_posterior_unnorm, dim=-1, keepdim=True)  # shape: (B, M, 1)
            log_posterior = log_posterior_unnorm - log_sumexp  # shape: (B, M, F) - (B, M, 1) = (B, M, F)

            # Compute the posterior probabilities without in-place operations
            post_pr = torch.exp(log_posterior)  # shape: (B, M, F)
            post_pr = post_pr.to(gt_trajs.device)

        # Compute loss.
        loss = 0.0
        for mode in range(num_modes):
            nll_k = self.nll_pytorch_dist(pred_trajs[:, mode], gt_trajs, gt_valid_mask, return_loss=True)  # shape: (B)
            loss_k = nll_k.unsqueeze(1) * post_pr[:, mode]  # shape: (B, 1) * (B, F) = (B, F)
            loss += loss_k.mean()

        # Adding entropy loss term to ensure that individual predictions do not try to cover multiple modes.
        entropy_vals = [
            self.compute_bivariate_gaussian_distribution(pred_trajs[:, mode]).entropy() for mode in range(num_modes)
        ]
        entropy_vals = torch.stack(entropy_vals).permute(1, 0, 2)  # shape: (M, B, F) -> (B, M, F)
        entropy_loss = torch.mean((entropy_vals).sum(2).max(1)[0])  # shape: (B, M) -> (B) -> 1
        loss += self.entropy_weight * entropy_loss

        # KL divergence between the prior and the posterior distributions.
        kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
        kl_loss = self.kl_weight * kl_loss_fn(torch.log(pred_scores), post_pr.sum(dim=2))  # shape: (B, M) -> (B) -> 1

        # compute ADE/FDE loss - L2 norms with between best predictions and GT.
        if self.use_fdeade_aux_loss:
            adefde_loss = self.l2_loss_fde(pred_trajs, gt_trajs, gt_valid_mask)
        else:
            adefde_loss = torch.tensor(0.0).to(gt_trajs.device)

        # post_entropy
        return loss + kl_loss + adefde_loss

    def compute_bivariate_gaussian_distribution(self, pred: torch.Tensor) -> MultivariateNormal:
        """Get the bivariate Gaussian distributions for the predicted trajectories of a single mode.

        Notation:
            B: batch_size
            F: number of future steps
            D = 5: gmm dimensions

        Args:
            pred (torch.Tensor): predicted trajectories of shape (B, F, D=5)

        Returns:
            biv_gauss_dist (MultivariateNormal): a PyTorch distribution object representing the bivariate Gaussian
                distributions for each predicted trajectory.
        """
        assert pred.shape[-1] == self.nonsquare_gmm_size, (
            f"Expected last dimension of pred to be {self.nonsquare_gmm_size}, but got {pred.shape[-1]}"
        )

        mu_x = pred[:, :, 0].unsqueeze(2)  # shape: (B, F, 1)
        mu_y = pred[:, :, 1].unsqueeze(2)  # shape: (B, F, 1)
        sigma_x = pred[:, :, 2]  # shape: (B, F)
        sigma_y = pred[:, :, 3]  # shape: (B, F)
        rho = pred[:, :, 4]  # shape: (B, F)

        # Create the base covariance matrix for a single element
        cov = torch.stack(
            [
                torch.stack([sigma_x**2, rho * sigma_x * sigma_y], dim=-1),  # shape: (B, F, 2)
                torch.stack([rho * sigma_x * sigma_y, sigma_y**2], dim=-1),  # shape: (B, F, 2)
            ],
            dim=-2,
        )  # shape: (B, F, 2, 2)

        # Expand this base matrix to match the desired shape
        return MultivariateNormal(loc=torch.cat((mu_x, mu_y), dim=-1), covariance_matrix=cov, validate_args=False)

    def compute_laplace_distribution(self, pred: torch.Tensor) -> Laplace:
        """Get the bivariate Laplace distributions for the predicted trajectories.

        Args:
            pred (torch.Tensor): predicted trajectory of shape (B, F, D=5)

        Returns:
            laplace_dist (Laplace): a PyTorch distribution object representing the bivariate Laplace.
        """
        # laplace(loc=mu, scale=sigma)
        return Laplace(loc=pred[..., :2], scale=pred[..., 2:4], validate_args=False)

    def nll_pytorch_dist(
        self, pred: torch.Tensor, gt: torch.Tensor, gt_valid_mask: torch.Tensor, *, return_loss: bool = True
    ) -> torch.Tensor:
        """Compute the negative log-likelihood of the data under the predicted distribution.

        Args:
            pred (torch.Tensor): predicted trajectories for one mode of shape (B, F, D=5)
            gt (torch.Tensor): ground-truth trajectories of shape (B, F, D=3), where only x and y are used.
            gt_valid_mask (torch.Tensor): binary mask of shape (B, F) indicating valid ground truth points
            return_loss (bool): if True, returns the mean negative log-likelihood loss; if
                False, returns the negative log-likelihood for each element in the batch without averaging.

        Returns:
            nll (torch.Tensor): the negative log-likelihood of the data under the predicted distribution.
        """
        laplace_dist = self.compute_laplace_distribution(pred)
        gt_xy = gt[:, :, :2]
        # Return the mean negative log-likelihood loss if return_loss is True, otherwise return the negative
        # log-likelihood for each element in the batch without averaging.
        if return_loss:
            return ((-laplace_dist.log_prob(gt_xy)).sum(-1) * gt_valid_mask).sum(1)  # shape: (B)
        return (-laplace_dist.log_prob(gt_xy)).sum(dim=2) * gt_valid_mask  # shape: (B, F)

    def l2_loss_fde(self, pred: torch.Tensor, gt: torch.Tensor, gt_valid_mask: torch.Tensor) -> torch.Tensor:
        """Compute the L2 loss for the final displacement error (FDE) and average displacement error (ADE) between the
        predicted trajectories and the ground truth trajectories.

        Args:
            pred (torch.Tensor): predicted trajectories for one mode of shape (B, M, F, D=5)
            gt (torch.Tensor): ground truth trajectories of shape (B, F, D=3)
            gt_valid_mask (torch.Tensor): binary mask of shape (B, F) indicating valid ground truth points.

        Returns:
            loss (torch.Tensor): the computed L2 loss for FDE and ADE.
        """
        diff = pred[:, :, :, :2] - gt[:, None, :, :2]  # shape: (B, M, F, 2)
        fde_loss = torch.norm(diff[:, :, -1], p=2, dim=-1) * gt_valid_mask[:, None, -1]
        ade_loss = (torch.norm(diff.transpose(1, 2), p=2, dim=-1) * gt_valid_mask.unsqueeze(-1)).mean(dim=1)
        loss, _ = (fde_loss + ade_loss).min(dim=1)
        return self.fdeade_weight * loss.mean()
