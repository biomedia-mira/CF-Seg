import os
import sys
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from layers import TraceStorage_ELBO
from matplotlib import colors
from torch import Tensor, nn

sys.path.append("..")
# from hps import Hparams


mimic_var_shape = {"sex":2,         # male, female
                   "view":2,        # PA, AP
                   "finding":4,     # NoFinding, PE, CM, PE&CM
                   "pe_finding":2,  # NoFinding, PE
                   "age":1,         # 0-100
                   "race":3,}       # White, Asian, Black



mimic_var_values = {"sex":[0,1],         # male, female
                   "view":[0,1],         # AP, PA
                   "finding":[0,1,2,3],  # NoFinding, PE, CM, PE&CM
                   "pe_finding":[0,1],   # NoFinding, PE
                   "age":[0,1,2,3,4],    # 0-100
                   "race":[0,1,2],}      # White, Asian, Black


def get_title_mimic(do, j):
    msg = ""
    for i, (k, v) in enumerate(do.items()):
        if k == "sex":
            sex_categories = ["male", "female"]  # 0,1
            vv = sex_categories[int(torch.argmax(v[j], dim=-1))]
            kk = "s"
        elif k == "view":
            view_categories = ["AP", "PA"]  # 0,1
            vv = view_categories[int(torch.argmax(v[j], dim=-1))]
            kk = "v"
        elif k == "race":
            race_categories = ["white", "asian", "black"]  # 0,1,2
            vv = race_categories[int(torch.argmax(v[j], dim=-1))]
            kk = "r"
        elif k == "finding":
            finding_categories = ["NoFinding", "PE", "CM", "PE&CM"]  # 0,1,2,3
            vv = finding_categories[int(torch.argmax(v[j], dim=-1))]
            kk = "f"
        elif k == "pe_finding":
            finding_categories = ["NoFinding", "PE"]  # 0,1
            vv = finding_categories[int(torch.argmax(v[j], dim=-1))]
            kk = "fp"
        elif k == "age":
            vv = str(int(v[j].item()))
            kk = "a"
        else:
            NotImplementedError
        msg += kk + "=" + vv
        # msg += kk + "{{=}}" + vv
        msg += "\n" if (i + 1) < len(list(do.keys())) else ""
    return msg


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max([np.abs(self.vmin), np.abs(self.vmax)])
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def check_nan(input_dict: Dict[str, Tensor]):
    nans = 0
    for k, v in input_dict.items():
        k_nans = torch.isnan(v).sum()
        nans += k_nans
        if k_nans > 0:
            print(f"\nFound {k_nans} nan(s) in {k}, skipping step.")
    return nans


def update_stats(stats: Dict[str, Any], elbo_fn: TraceStorage_ELBO):
    """Accumulate tracked summary statistics."""

    def _update(trace, dist="p"):
        for name, node in trace.nodes.items():
            if node["type"] == "sample":
                k = "log" + dist + "(" + name + ")"
                if k not in stats:
                    stats[k] = 0
                stats[k] += node["log_prob"].sum().item()
        return stats

    _update(elbo_fn.trace_storage["model"], dist="p")
    _update(elbo_fn.trace_storage["guide"], dist="q")
    return stats


def plot(x, fig=None, ax=None, nrows=1, cmap="Greys_r", norm=None, cbar=False, set_cbar_ticks=True,):
    m, n = nrows, x.shape[0] // nrows
    if ax is None:
        fig, ax = plt.subplots(m, n, figsize=(n * 4, 8))
    im = []
    for i in range(m):
        for j in range(n):
            idx = (i, j) if m > 1 else j
            ax = [ax] if n == 1 else ax
            _x = x[i * n + j].squeeze()
            if norm is not None:
                norm = MidpointNormalize(vmin=_x.min(), midpoint=0, vmax=_x.max())
            _im = ax[idx].imshow(_x, cmap=cmap, norm=norm)
            im.append(_im)
            ax[idx].axes.xaxis.set_ticks([])
            ax[idx].axes.yaxis.set_ticks([])
    if cbar:
        if fig:
            fig.subplots_adjust(wspace=-0.275, hspace=0.25)
        for i in range(m):
            for j in range(n):
                idx = [i, j] if m > 1 else j
                cbar_ax = fig.add_axes([ax[idx].get_position().x0, ax[idx].get_position().y0 - 0.015, ax[idx].get_position().width, 0.0075,])
                cbar = plt.colorbar(im[i * n + j], cax=cbar_ax, orientation="horizontal")  
                _x = x[i * n + j].squeeze()
                if set_cbar_ticks:
                    d = 20
                    _vmin, _vmax = _x.min().abs().item(), _x.max().item()
                    _vmin = -(_vmin - (_vmin % d))
                    _vmax = _vmax - (_vmax % d)
                    lt = [_vmin, 0, _vmax]
                    if (np.abs(_vmin) - 0) > d or (_vmax - 0) > d:
                        lt.insert(1, _vmin // 2)
                        lt.insert(-2, _vmax // 2)
                    cbar.set_ticks(lt)
                else:
                    cbar.ax.locator_params(nbins=5)
                    cbar.formatter.set_powerlimits((0, 0))
                cbar.outline.set_visible(False)
    return fig, ax


@torch.no_grad()
def plot_cf(x: Tensor, cf_x: Tensor, pa: Dict[str, Tensor], cf_pa: Dict[str, Tensor], do: Dict[str, Tensor], var_cf_x: Optional[Tensor], num_images: int = 8,):
    n = num_images  # 8 columns
    x = (x[:n].detach().cpu() + 1) * 127.5
    cf_x = (cf_x[:n].detach().cpu() + 1) * 127.5

    fs = 24  # font size
    pad = 8
    m = 3 if var_cf_x is None else 4  # nrows
    s = 5
    fig, ax = plt.subplots(m, n, figsize=(n*s, m*s), facecolor="white")

    _, _ = plot(x, ax=ax[0])
    _, _ = plot(cf_x, ax=ax[1])
    _, _ = plot(cf_x - x, ax=ax[2], fig=fig, cmap="RdBu_r", cbar=True, norm=MidpointNormalize(midpoint=0),)
    if var_cf_x is not None:
        _, _ = plot(var_cf_x[:n].clamp(min=0).detach().sqrt().cpu(), fig=fig, cmap="jet", ax=ax[3], cbar=True, set_cbar_ticks=False,)

    for j in range(n):
        pa_msg = get_title_mimic(pa, j)
        do_msg = get_title_mimic(do, j)
        cf_pa_msg = get_title_mimic(cf_pa, j)
        ax[0, j].set_title(rf"{pa_msg}", pad=pad+2, fontsize=fs-8, multialignment="center", linespacing=1.5)
        ax[1, j].set_title(rf"do(${do_msg}$)", fontsize=fs-2, pad=pad+4)
        ax[1, j].set_xlabel(rf"{cf_pa_msg}", labelpad=pad+4, fontsize=fs-8, multialignment="center",linespacing=1.25)

    ax[0, 0].set_ylabel("Observation", fontsize=fs + 4, labelpad=pad)
    ax[1, 0].set_ylabel("Counterfactual", fontsize=fs + 4, labelpad=pad)
    ax[2, 0].set_ylabel("Direct Effect", fontsize=fs + 4, labelpad=pad)
    if var_cf_x is not None:
        ax[3, 0].set_ylabel("Uncertainty", fontsize=fs + 4, labelpad=pad)
    return fig




def calculate_loss(pred_batch, target_batch, loss_norm="l1", soft_loss="BCElogits"):
    "Calculate the losses for pred_bacth"
    loss=0
    for k in pred_batch.keys():
        assert pred_batch[k].size()==target_batch[k].size(), f"{k} size does not match, pred_batch size {pred_batch[k].size()}; target batch size {target_batch[k].size()}"
        if k=="age":
            if loss_norm=="l1":
                loss+=torch.nn.L1Loss()(pred_batch[k], target_batch[k]) 
            elif loss_norm=="l2":
                loss+=torch.nn.MSELoss()(pred_batch[k], target_batch[k]) 
        elif k in ["sex", "finding", "race", "view", "pe_finding"]:
            if soft_loss=="BCElogits":
                loss+=torch.nn.CrossEntropyLoss()(pred_batch[k], target_batch[k])
            elif soft_loss=="l1":
                loss+=torch.nn.L1Loss()(pred_batch[k], target_batch[k]) 
        else:
            NotImplementedError
    return loss