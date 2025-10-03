import math
from collections import OrderedDict

import torch
from easydict import EasyDict
from einops import rearrange
from torch import nn
from torch.nn import functional as F


def initialize_weights_with_normal(module: nn.Module, init_scale: float = 0.02) -> None:
    """Weight initialization using nn.init.normal_.

    Args:
        module (nn.Module): an architecture, instance of nn.Module.
        init_scale (float): optional scaling factor.
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=init_scale)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(mean=0.0, std=init_scale)


def initialize_weights_with_xavier(module: nn.Linear | nn.Embedding, init_gain: float = 1.0) -> None:
    """Weight initialization using nn.init.xavier_normal_.

    Args:
        module (nn.Module): an architecture, instance of nn.Module.
        init_gain (float): optional scaling factor.
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=init_gain)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.xavier_normal_(m.weight, gain=init_gain)


class ModuleOutput(OrderedDict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


class Residual(nn.Module):
    def __init__(self, module: nn.Module, dropout: float = 0.0):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(dropout)

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        output.last_hidden_state = self.dropout(output.last_hidden_state) + args[0]
        return output


class RotaryPositionEmbedding:
    # Specified in https://arxiv.org/abs/2104.09864
    # Modified from https://github.com/lucidrains/rotary-embedding-torch
    def __init__(self, frq_pos_enc: torch.Tensor, right_align: bool = False):
        # frq_pos_enc shape is (b, n, c).
        # frq_pos_enc is broadcast to (b, h, n, c).
        self.frq_pos_enc = rearrange(frq_pos_enc, "b n c -> b 1 n c")
        self.rotate_dim = frq_pos_enc.shape[-1]
        self.right_align = right_align

    def rotate(self, t):
        seq_len = t.shape[-2]
        if self.right_align:
            # q and k are right-aligned in Perceiver AR
            pos_enc = self.frq_pos_enc[..., -seq_len:, :]
        else:
            # q and k are left-aligned
            pos_enc = self.frq_pos_enc[..., :seq_len, :]

        t_rot, t_pass = t[..., : self.rotate_dim], t[..., self.rotate_dim :]
        t_rot = (t_rot * pos_enc.cos()) + (self._rotate_half(t_rot) * pos_enc.sin())

        return torch.cat((t_rot, t_pass), dim=-1)

    @staticmethod
    def _rotate_half(x):
        # Rearranges channel dimension [x1, x2, x3, x4, ...] -> [-x2, x1, -x4, x3, ...]
        x = rearrange(x, "... (c r) -> ... c r", r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, "... c r -> ... (c r)")


class MLP(nn.Sequential):
    """A simple MultiLayerPerceptron (MLP) wrapper with GELU activation."""

    def __init__(self, num_channels: int, widening_factor: int, bias: bool = True):
        """MLP initialization module.

        Args:
            num_chanels (int): number of input channels
            widening_factor (int): factor by which to widen the embedding space.
            bias (bool): if True will add a bias to the MLP.
        """
        super().__init__(
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, widening_factor * num_channels, bias=bias),
            nn.GELU(),
            nn.Linear(widening_factor * num_channels, num_channels, bias=bias),
        )

    def forward(self, x: torch.Tensor, return_dict: bool = True) -> ModuleOutput | torch.Tensor:
        """Model's forward function.

        Args:
            x[torch.Tensor(B, A, T, C)]: input tensor to be decoded.
                B: batch size
                A: number of agents
                T: trajectory length
                C: number of input dimensions.

        Returns:
            x[torch.Tensor(B, A, T, H)]: extracted feature.
        """
        x = super().forward(x)
        if return_dict:
            return ModuleOutput(last_hidden_state=x)
        return x



class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim: int, bias: bool, eps: float = 1e-5) -> None:
        """LayerNorm initialization.

        Inputs
        ------
            ndim[int]: number of input dimensions.
            bias[bool]: whether to add a bias.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Model's forward function.

        Args:
            x[torch.Tensor(B, A, T, C)]: input tensor to be decoded.
                B: batch size
                A: number of agents
                T: trajectory length
                C: number of input dimensions.

        Returns:
            x[torch.Tensor(B, A, T, H)]: extracted feature.
        """
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)


class ContextNet(nn.Module):
    """ Context module used to extract polyline features of the map. It simply consists of MLP + BN
    blocks and it does not implement feature aggregation. """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_vectors: int,
        reduction_factor: int = 16
    ) -> None:
        super().__init__()
        self.pointnet = [
            nn.Linear(input_size, hidden_size, bias=True),
            nn.BatchNorm1d(num_vectors),
            nn.ReLU()
        ]
        self.pointnet = nn.Sequential(*self.pointnet)

        self.pointnet_out = [
            nn.Linear(num_vectors * hidden_size, num_vectors * (hidden_size // reduction_factor), bias=True),
            nn.BatchNorm1d(num_vectors * (hidden_size // reduction_factor)),
            nn.ReLU(),
            nn.Linear(num_vectors * (hidden_size // reduction_factor), hidden_size)
        ]
        self.pointnet_out = nn.Sequential(*self.pointnet_out)

    def forward(self, x):
        # B, N, P, D
        B, P, N, D = x.size()
        x = x.transpose(1, 2).reshape(B * N, P, D)
        x = self.pointnet(x)

        BN, _, E = x.size()
        x = x.reshape(BN, -1)
        x = self.pointnet_out(x)
        x = x.reshape(B, N, E)
        return x
