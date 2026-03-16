import sys

sys.path.append("..")
from typing import Dict

import torch
from layers import TraceStorage_ELBO
from torch import Tensor, nn
from utils_pgm import check_nan, calculate_loss

from hps import Hparams

import torch.nn.functional as F

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



class DSCM(nn.Module):
    def __init__(self, args: Hparams, pgm: nn.Module, predictor: nn.Module, vae: nn.Module):
        super().__init__()
        self.args = args
        self.pgm = pgm  # DAG model excluding x
        self.pgm.eval()
        self.pgm.requires_grad = False
        self.predictor = predictor  # parent classifiers
        self.predictor.eval()
        self.predictor.requires_grad = False
        self.vae = vae  # HVAE for x
        # lagrange multiplier
        self.lmbda = nn.Parameter(args.lmbda_init * torch.ones(1))
        self.register_buffer("eps", args.elbo_constraint * torch.ones(1))

    def forward(self, obs: Dict[str, Tensor], do: Dict[str, Tensor], elbo_fn: TraceStorage_ELBO, cf_particles: int = 1, t_abduct: float = 1.0,) -> Dict[str, Tensor]:

        pa = {k: v for k, v in obs.items() if k not in ["x", "y", "dicom_id"]}
        _pa = vae_preprocess(self.args, {k: v.clone() for k, v in pa.items()})

        # forward vae with factual parents
        vae_out = self.vae(obs["x"], _pa, beta=self.args.beta)

        # Get soft labels, should be normalized values
        with torch.no_grad():
            soft_labels = self.predictor.predict(**obs)

        if cf_particles > 1:  # for calculating counterfactual uncertainty
            cfs = {"x": torch.zeros_like(obs["x"])}
            cfs.update({"x2": torch.zeros_like(obs["x"])})

        for _ in range(cf_particles):
            # forward pgm, get counterfactual parents
            cf_pa = self.pgm.counterfactual(obs=pa, intervention=do, num_particles=1)
            _cf_pa = vae_preprocess(self.args, {k: v.clone() for k, v in cf_pa.items()})
            
            # forward vae with counterfactual parents
            zs = self.vae.abduct(obs["x"], parents=_pa, t=t_abduct)  # z ~ q(z|x,pa)

            if self.vae.cond_prior:
                zs = [zs[j]["z"] for j in range(len(zs))]

            rec_loc, rec_scale = self.vae.forward_latents(zs, parents=_pa)
            cf_loc, cf_scale = self.vae.forward_latents(zs, parents=_cf_pa)
            u = (obs["x"] - rec_loc) / rec_scale.clamp(min=1e-12)
            cf_x = torch.clamp(cf_loc + cf_scale * u, min=-1, max=1)

            if cf_particles > 1:
                cfs["x"] += cf_x
                with torch.no_grad():
                    cfs["x2"] += cf_x**2
            else:
                cfs = {"x": cf_x}

        # Var[X] = E[X^2] - E[X]^2
        if cf_particles > 1:
            with torch.no_grad():
                var_cf_x = (cfs["x2"] - cfs["x"] ** 2 / cf_particles) / cf_particles
                cfs.pop("x2", None)
            cfs["x"] = cfs["x"] / cf_particles
        else:
            var_cf_x = None

        cfs.update(cf_pa)
        if check_nan(vae_out) > 0 or check_nan(cfs) > 0:
            return {"loss": torch.tensor(float("nan"))}

        if self.args.soft_cf:
            # use cfs directly for continuous variables of pgm
            for k in self.pgm.variables.keys():
                if self.pgm.variables[k] == "continuous":
                    soft_labels[k] = cfs[k]

            for do_key in do.keys():

                for child in self.pgm.graph[do_key]: # get childeren of do_key from pgm graph
                    # we only consider graph variables. 
                    # This is to avoide "x", but it can be generalizable in future.
                    if child in self.pgm.graph.keys():
                        soft_labels[child] = cfs[child] 

                # Set soft_labels[do_key] as the intervened label
                soft_labels[do_key] = cfs[do_key] * self.args.do_multip


            # Use other sup loss, here target batch should be the soft labels
            pred_batch = self.predictor.predict(**cfs)
            aux_loss = calculate_loss(pred_batch=pred_batch, target_batch=soft_labels, loss_norm="l2") 
        else:
            # we are using predicted counterfactual variable values as ground truth.
            # In the PGM generated counterfacutal, values for variables, such as 
            # view and finding, can be continous because of softmax_centered trick.
            # This raises error because they are not one hot encoded (necessity for predictors).
            # As such we convert continous softmax_centered values to one-hot encoded    
            for k in cfs.keys():
                if k in self.args.dummy_var:
                    temp = torch.argmax(cfs[k], dim=-1)
                    n_classes = mimic_var_shape[k]
                    if self.args.add_dummy_dim:
                        n_classes += 1
                    cfs[k] = F.one_hot(temp, num_classes=n_classes)
            aux_loss = (elbo_fn.differentiable_loss(self.predictor.model_anticausal, self.predictor.guide_pass, **cfs) / cfs["x"].shape[0])

        with torch.no_grad():
            sg = self.eps - vae_out["elbo"]
        damp = self.args.damping * sg
        loss = aux_loss - (self.lmbda - damp) * (self.eps - vae_out["elbo"])

        out = {}
        out.update(vae_out)
        out.update({"loss": loss, "aux_loss": aux_loss, "cfs": cfs, "var_cf_x": var_cf_x})
        return out



def vae_preprocess(args: Hparams, pa: Dict[str, Tensor]) -> Tensor:
    # concatenate parents, expand to input res for conditioning the vae
    concat_pa = torch.cat([pa[k] if len(pa[k].shape) > 1 else pa[k][..., None] for k in args.parents_x], dim=1,)
    concat_pa = (concat_pa[..., None, None].repeat(1, 1, *(args.input_res,) * 2).cuda().float())
    return concat_pa


if __name__ == "__main__":
    # test
    args = Hparams()
    args.dataset = "none"
    args.input_res = 28
    args.parents_x = ["a", "b", "c"]
    pa = {"a": torch.ones(2, 1), "b": torch.ones(2, 1), "c": torch.ones(2, 1)}
    out = vae_preprocess(args, pa)
    assert out.shape == (2, 3, 28, 28)
