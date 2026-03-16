import pyro
import torch
import torch.nn as nn
from pyro.distributions.torch_distribution import TorchDistributionMixin
from pyro.distributions.conditional import *
from typing import List, Dict, Any
from torch.distributions import constraints
from torch.distributions.utils import _sum_rightmost
from torch.distributions.transforms import  Transform
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from pyro.distributions.torch import Gumbel
from pyro.distributions.torch import Categorical
from pyro.distributions.torch_distribution import ExpandedDistribution

class TraceStorage_ELBO(pyro.infer.Trace_ELBO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.trace_storage = {'model': None, 'guide': None}

    def _get_trace(self, model, guide, args, kwargs):
        model_trace, guide_trace = super()._get_trace(model, guide, args, kwargs)

        self.trace_storage['model'] = model_trace
        self.trace_storage['guide'] = guide_trace

        return model_trace, guide_trace


class ConditionalAffineTransform(ConditionalTransformModule):
    def __init__(self, context_nn: nn.Module, event_dim: int = 0, **kwargs: Any):
        super().__init__(**kwargs)
        self.event_dim = event_dim
        self.context_nn = context_nn

    def condition(self, context: Tensor):
        loc, log_scale = self.context_nn(context)
        return torch.distributions.transforms.AffineTransform(loc, F.softplus(log_scale), event_dim=self.event_dim)


class SoftmaxCentered(Transform):
    """
    Implements softmax as a bijection, the forward transformation appends a value to the
    input and the inverse removes it. The appended coordinate represents a pivot, e.g., 
    softmax(x) = exp(x-c) / sum(exp(x-c)) where c is the implicit last coordinate.

    Adapted from a Tensorflow implementation: https://tinyurl.com/48vuh7yw 
    """
    domain = constraints.real_vector
    codomain = constraints.simplex

    def __init__(self, temperature: float = 1.):
        super().__init__()
        self.temperature = temperature

    def __call__(self, x: Tensor): # x is an output of a NN
        zero_pad = torch.zeros(*x.shape[:-1], 1, device=x.device) 
        x_padded = torch.cat([x, zero_pad], dim=-1)           # add extra dimension: ex. 2 -> 3
        return (x_padded / self.temperature).softmax(dim=-1)

    def _inverse(self, y: Tensor):
        log_y = torch.log(y.clamp(min=1e-12))
        unorm_log_probs = log_y[..., :-1] - log_y[..., -1:] # remove the extra dimension ex. 3 -> 2
        # unorm_log_probs[unorm_log_probs<=1e-12] = 0.0
        return unorm_log_probs * self.temperature

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor):
        """ log|det(dy/dx)| """
        Kplus1 = torch.tensor(y.size(-1), dtype=y.dtype, device=y.device)
        return 0.5 * Kplus1.log() + torch.sum(torch.log(y.clamp(min=1e-12)), dim=-1)

    def forward_shape(self, shape: torch.Size):
        return shape[:-1] + (shape[-1] + 1,)  # forward appends one dim

    def inverse_shape(self, shape: torch.Size):
        if shape[-1] <= 1:
            raise ValueError
        return shape[:-1] + (shape[-1] - 1,)  # inverse removes last dim