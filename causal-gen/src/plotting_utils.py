import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colors
# from pgm.utils_pgm import get_title_mimic

mimic_var_shape_dummy = {"sex":2,
                         "view":2+1,
                         "finding":4+1,
                         "pe_finding":2+1,
                         "age":1,
                         "race":3,}

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


def replace_value(val, possible_values):
    """Randomly select a value from the possible values excluding the current value."""
    possible_values = [x for x in possible_values if x != val]
    return possible_values[torch.randint(len(possible_values), (1,)).item()]



def plot_cxr_grid(x, fig=None, ax=None, nrows=1, cmap="Greys_r", norm=None, cbar=False):
    m, n = nrows, x.shape[0] // nrows
    if ax is None:
        fig, ax = plt.subplots(m, n, figsize=(n * 4, 8))
    im = []
    for i in range(m):
        for j in range(n):
            idx = (i, j) if m > 1 else j
            ax = [ax] if n == 1 else ax
            _x = x[i * n + j].squeeze()
            if _x.shape[0] == 3:
                _x = np.transpose(_x, [1, 2, 0]).int()
            if norm is not None:
                norm = MidpointNormalize(vmin=_x.min(), midpoint=0, vmax=_x.max())
            _im = ax[idx].imshow(_x, cmap=cmap, norm=norm)
            im.append(_im)
            ax[idx].axes.xaxis.set_ticks([])
            ax[idx].axes.yaxis.set_ticks([])

    # plt.tight_layout()

    if cbar:
        if fig:
            fig.subplots_adjust(wspace=-0.3, hspace=0.3)
        for i in range(m):
            for j in range(n):
                idx = [i, j] if m > 1 else j

                cbar_ax = fig.add_axes([ax[idx].get_position().x0, ax[idx].get_position().y0 - 0.02, ax[idx].get_position().width, 0.01,])
                cbar = plt.colorbar(im[i * n + j], cax=cbar_ax, orientation="horizontal")

                _x = x[i * n + j].squeeze()

                d = 20
                _vmin, _vmax = _x.min().abs().item(), _x.max().item()
                _vmin = -(_vmin - (_vmin % d))
                _vmax = _vmax - (_vmax % d)

                lt = [_vmin, 0, _vmax]

                if (np.abs(_vmin) - 0) > d:
                    lt.insert(1, _vmin // 2)
                if (_vmax - 0) > d:
                    lt.insert(-2, _vmax // 2)

                cbar.set_ticks(lt)
                cbar.outline.set_visible(False)

    return fig, ax


def undo_norm(pa):
    # reverse [-1,1] parent preprocessing back to original range
    for k, v in pa.items():
        if k == "age":
            pa[k] = (v + 1) / 2 * 100  # [-1,1] -> [0,100]
    return pa


@torch.no_grad()
def plot_counterfactual_viz_mimic(args, x, cf_x, pa, cf_pa, do, rec_loc, figname, save=True):

    pa = undo_norm(pa)
    cf_pa = undo_norm(cf_pa)
    do = undo_norm(do)

    fs = 15
    m, s = 6, 3
    n = 8
    fig, ax = plt.subplots(m, n, figsize=(n * s - 2, m * s))

    x = (x[:n].detach().cpu()+1) * 255 / 2
    cf_x = (cf_x[:n].detach().cpu()+1) * 255 / 2
    rec_loc = (rec_loc[:n].detach().cpu()+1) * 255 / 2

    # this is required as mimic has RGB channels and it leads to some issue during plotting.
    x = x[:,0,...]
    cf_x = cf_x[:,0,...]
    rec_loc = rec_loc[:,0,...]

    _, _ = plot_cxr_grid(x,              ax=ax[0])
    _, _ = plot_cxr_grid(rec_loc,        ax=ax[1])
    _, _ = plot_cxr_grid(cf_x,           ax=ax[2])
    _, _ = plot_cxr_grid(rec_loc - x,    ax=ax[3], fig=fig, cmap="RdBu_r", cbar=True, norm=MidpointNormalize(midpoint=0),)
    _, _ = plot_cxr_grid(cf_x - x,       ax=ax[4], fig=fig, cmap="RdBu_r", cbar=True, norm=MidpointNormalize(midpoint=0),)
    _, _ = plot_cxr_grid(cf_x - rec_loc, ax=ax[5], fig=fig, cmap="RdBu_r", cbar=True, norm=MidpointNormalize(midpoint=0),)

    for j in range(n):
        title = get_title_mimic(pa, j)
        msg = get_title_mimic(do, j)
        ax[0, j].set_title(rf"{title}", fontsize=fs - 5, multialignment="center", linespacing=1.5,)
        ax[1, j].set_title("rec_loc", pad=8)
        ax[2, j].set_title(rf"do(${msg}$)", fontsize=fs - 2, pad=8)
        ax[3, j].set_title("rec_loc - x")
        ax[4, j].set_title("cf_loc - x", fontsize=fs - 5, multialignment="center", linespacing=1.5,)
        ax[5, j].set_title("cf_loc - rec_loc", pad=8)

    if save:
        fig.savefig(os.path.join(args.save_dir, f"viz-{args.iter}-{figname}.png"), bbox_inches="tight")
        plt.close()
        return

    return fig


def write_images(args, model, batch):

    # reconstructions, first abduct z from q(z|x,pa)
    zs = model.abduct(x=batch["x"], parents=batch["pa"])

    if model.cond_prior:
        zs = [zs[j]["z"] for j in range(len(zs))]

    if args.dataset in ["mimic","mimic_pe"]:

        pa = {k: batch[k] for k in args.parents_x}                 # parents
        _pa = torch.cat([batch[k] for k in args.parents_x], dim=1) # parents concatenated
        _pa = (_pa[..., None, None].repeat(1, 1, *(args.input_res,) * 2).to(args.device).float()) # parents to input res

        rec_loc, _ = model.forward_latents(zs, parents=_pa) # reconstruction

        # counterfactuals
        for k in args.parents_x:
            cf_pa = {k: batch[k] for k in args.parents_x}
            if k != "age":
                new_pa = torch.tensor([replace_value(val, list(range(mimic_var_shape[k]))) for val in torch.argmax(cf_pa[k],1)])
                # new_pa = torch.randint_like(torch.argmax(cf_pa[k],1), high=mimic_var_shape[k])
                shape = mimic_var_shape_dummy[k] if args.add_dummy_dim else mimic_var_shape[k]
                cf_pa[k] = F.one_hot(new_pa, num_classes=shape)
            else: 
                cf_pa[k] = torch.randint_like(cf_pa[k], high=100) / 100 * 2 -1 # [-1,1]
            do = {k: cf_pa[k]}
            _cf_pa = torch.cat([cf_pa[k] for k in args.parents_x], dim=1)
            _cf_pa = (_cf_pa[..., None, None].repeat(1, 1, *(args.input_res,) * 2).to(args.device).float())
            cf_loc, _ = model.forward_latents(zs, parents=_cf_pa)
            plot_counterfactual_viz_mimic(args, batch["x"], cf_loc, pa.copy(), cf_pa, do, rec_loc, f"change({k})")

        del rec_loc, cf_loc

        return 

    else:
        raise NotImplementedError