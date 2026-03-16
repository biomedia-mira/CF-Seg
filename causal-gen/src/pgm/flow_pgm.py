from typing import List, Dict, Any

import numpy as np
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import torch
import torch.nn.functional as F
from layers import (  # fmt: skip
    ConditionalAffineTransform,
    SoftmaxCentered,
)
from pyro.distributions.conditional import ConditionalTransformedDistribution
from pyro.infer.reparam.transform import TransformReparam
from pyro.nn import DenseNN
from torch import Tensor, nn
from resnet import CustomBlock, ResNet, ResNet18

from hps import Hparams


class BasePGM(nn.Module):
    def __init__(self):
        super().__init__()

    def scm(self, *args, **kwargs):
        def config(msg):
            if isinstance(msg["fn"], dist.TransformedDistribution):
                return TransformReparam()
            else:
                return None

        return pyro.poutine.reparam(self.model, config=config)(*args, **kwargs)

    def sample_scm(self, n_samples: int = 1):
        with pyro.plate("obs", n_samples):
            samples = self.scm()
        return samples

    def sample(self, n_samples: int = 1):
        with pyro.plate("obs", n_samples):
            samples = self.model()  # NOTE: not ideal as model is defined in child class
        return samples

    def infer_exogeneous(self, obs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        batch_size = list(obs.values())[0].shape[0]
        # assuming that we use transformed distributions for everything:
        cond_model = pyro.condition(self.sample, data=obs)
        cond_trace = pyro.poutine.trace(cond_model).get_trace(batch_size)

        output = {}
        for name, node in cond_trace.nodes.items():
            if "z" in name or "fn" not in node.keys():
                continue
            fn = node["fn"]
            if isinstance(fn, dist.Independent):
                fn = fn.base_dist
            if isinstance(fn, dist.TransformedDistribution):
                # compute exogenous base dist (created with TransformReparam) at all sites
                output[name + "_base"] = T.ComposeTransform(fn.transforms).inv(node["value"])
        return output

    def counterfactual(self, obs: Dict[str, Tensor], intervention: Dict[str, Tensor], num_particles: int = 1, detach: bool = True,) -> Dict[str, Tensor]:

        # NOTE: not ideal as "variables" is defined in child class
        dag_variables = self.variables.keys()
        assert set(obs.keys()) == set(dag_variables)
        avg_cfs = {k: torch.zeros_like(obs[k]) for k in obs.keys()}
        batch_size = list(obs.values())[0].shape[0]

        for _ in range(num_particles):
            # Abduction
            exo_noise = self.infer_exogeneous(obs)
            exo_noise = {k: v.detach() if detach else v for k, v in exo_noise.items()}
            # condition on root node variables (no exogeneous noise available)
            for k in dag_variables:
                if k not in intervention.keys():
                    if k not in [i.split("_base")[0] for i in exo_noise.keys()]:
                        exo_noise[k] = obs[k]
            # Abducted SCM
            abducted_scm = pyro.poutine.condition(self.sample_scm, data=exo_noise)
            # Action
            counterfactual_scm = pyro.poutine.do(abducted_scm, data=intervention)
            # Prediction
            counterfactuals = counterfactual_scm(batch_size)

            for k, v in counterfactuals.items():
                avg_cfs[k] += v / num_particles

        return avg_cfs


class ChestPGM(BasePGM):
    def __init__(self, args: Hparams, temperature: float = 1.0):

        super().__init__()

        self.variables = {
            "race": "categorical",    # 3 categories - Asian, White, Black
            "sex": "categorical",     # 2 categories - Male, Female
            "view": "categorical",    # 2 categories - PA, AP
            "finding": "categorical", # 4 categories - NoFinding, PE, CM, PE&CM
            "age": "continuous",      # 0 - 100
        }

        # parent: {childern}
        self.graph = {
            "race": {"x"},    
            "sex": {"x"},     
            "view": {"x"},    
            "finding": {"x","view"}, 
            "age": {"x","view","finding"},      
        }

        race_shape = 3
        sex_shape = 2
        view_shape = 2
        finding_shape = 4
        age_shape = 1

        # log space for sex, race, finding, and view
        self.sex_logits = nn.Parameter(np.log(1 / sex_shape) * torch.ones(1, sex_shape))
        self.race_logits = nn.Parameter(np.log(1 / race_shape) * torch.ones(1, race_shape))

        # define base distributions: 
        # age (continuous),
        self.a_base_loc = nn.Parameter(torch.zeros(age_shape))
        self.a_base_scale = nn.Parameter(torch.zeros(age_shape))  
        # finding (categorical with condition - Gumbel),
        self.register_buffer("f_base_loc", torch.zeros(finding_shape))
        self.register_buffer("f_base_scale", torch.ones(finding_shape))
        # view (categorical with condition - Gumbel)
        self.register_buffer("v_base_loc", torch.zeros(view_shape))
        self.register_buffer("v_base_scale", torch.ones(view_shape))

        # age spline flow
        self.age_flow_components = T.ComposeTransformModule([T.Spline(age_shape, count_bins=5, order="linear")])
        self.age_flow = T.ComposeTransform([self.age_flow_components,])

        # age -> finding : finding - gumble - loc and scale
        finding_net = DenseNN(age_shape, [8,16], param_dims=[finding_shape, finding_shape], nonlinearity=nn.ReLU())
        self.finding_flow = ConditionalAffineTransform(finding_net, event_dim=1)

        # [age,finding] -> view : view - gumble - loc and scale
        # adding finding + 1 as gumble softmax centered trick adds one dummy dimension
        view_net = DenseNN(age_shape + finding_shape + 1, [8,16], param_dims=[view_shape, view_shape], nonlinearity=nn.ReLU())        
        self.view_flow = ConditionalAffineTransform(view_net, event_dim=1)

        # initialize softmax transform with temperature
        self.softmax_transform = SoftmaxCentered(temperature)

        ####################################
        # define anticausal predictors

        shared_model = ResNet(CustomBlock, layers=[2, 2, 2, 2], widths=[64, 128, 256, 512], norm_layer=lambda c: nn.GroupNorm(min(32, c // 4), c),)
        shared_model.conv1 = nn.Conv2d(args.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False,)
        kwargs = {"in_shape": (args.input_channels, *(args.input_res,) * 2), "base_model": shared_model,}

        # q(s | x) ~ OneHotCategorical(f(x))
        self.encoder_s = ResNet18(num_outputs=sex_shape, **kwargs)

        # q(v | x) ~ OneHotCategorical(f(x)) 
        # additional shape for view due to softmax-centered trick
        self.encoder_v = ResNet18(num_outputs=view_shape+1, **kwargs)

        # q(r | x) ~ OneHotCategorical(f(x))
        self.encoder_r = ResNet18(num_outputs=race_shape, **kwargs)

        # q(f | x, v) ~ OneHotCategorical(f(x)) 
        # additional shape for finding due to softmax-centered trick
        # additional shape for view due to softmax-centered trick
        self.encoder_f = ResNet18(num_outputs=finding_shape+1, context_dim=view_shape+1, **kwargs)

        # q(a | x, f, v) ~ Normal(mu(x), sigma(x))
        # additional shape for finding due to softmax-centered trick
        # additional shape for view due to softmax-centered trick
        self.encoder_a = ResNet18(num_outputs=age_shape+age_shape, context_dim=view_shape+1+finding_shape+1, **kwargs)

    def model(self) -> Dict[str, Tensor]:

        pyro.module("ChestPGM", self)

        # p(r)
        pr = dist.OneHotCategorical(logits=self.race_logits).to_event(1)
        race = pyro.sample("race", pr)

        # p(s)
        ps = dist.OneHotCategorical(logits=self.sex_logits).to_event(1)
        sex = pyro.sample("sex", ps)

        # p(a)
        pa_base = dist.Normal(self.a_base_loc, F.softplus(self.a_base_scale, beta=np.log(2))).to_event(1)
        pa = dist.TransformedDistribution(pa_base, self.age_flow)
        age = pyro.sample("age", pa)

        # p(f|a)
        pf_a_base = dist.Gumbel(self.f_base_loc, self.f_base_scale).to_event(1)
        pf_a = ConditionalTransformedDistribution(pf_a_base, [self.finding_flow, self.softmax_transform],).condition(age)
        finding = pyro.sample("finding", pf_a)

        # p(v|f,a)
        pv_fa_base = dist.Gumbel(self.v_base_loc, self.v_base_scale).to_event(1)
        pv_fa = ConditionalTransformedDistribution(pv_fa_base, [self.view_flow, self.softmax_transform],).condition(torch.cat([age,finding],dim=1)) #because softmax centered adds a dummy dimension
        view = pyro.sample("view", pv_fa)

        return {
            "sex": sex,
            "race": race,
            "age": age,
            "finding": finding,
            "view": view,
        }

    def model_anticausal(self, **obs) -> None:
        # assumes all variables are observed, train classfiers
        pyro.module("ChestPGM", self)

        with pyro.plate("observations", obs["x"].shape[0]):

            # q(s | x)
            s_probs = F.softmax(self.encoder_s(obs["x"]), dim=-1)
            qs_x = dist.OneHotCategorical(probs=s_probs).to_event(1)
            pyro.sample("sex_aux", qs_x, obs=obs["sex"])

            # q(v | x)
            v_probs = F.softmax(self.encoder_v(obs["x"]), dim=-1)
            qv_x = dist.OneHotCategorical(probs=v_probs).to_event(1)
            pyro.sample("view_aux", qv_x, obs=obs["view"])

            # q(r | x)
            r_probs = F.softmax(self.encoder_r(obs["x"]), dim=-1)
            qr_x = dist.OneHotCategorical(probs=r_probs).to_event(1)
            pyro.sample("race_aux", qr_x, obs=obs["race"])

            # q(f | x, v)
            f_probs = F.softmax(self.encoder_f(obs["x"], y=obs["view"]), dim=-1) 
            qf_xv = dist.OneHotCategorical(probs=f_probs).to_event(1)
            pyro.sample("finding_aux", qf_xv, obs=obs["finding"])

            # q(a | x, f, v)
            a_loc, a_logscale = self.encoder_a(obs["x"], y=torch.cat([obs["finding"],obs["view"]],dim=1)).chunk(2, dim=-1) 
            qa_xfv = dist.Normal(a_loc, F.softplus(a_logscale, beta=np.log(2))).to_event(1)
            pyro.sample("age_aux", qa_xfv, obs=obs["age"])

    def predict(self, **obs) -> Dict[str, Tensor]:
        # q(s | x)
        s_probs = F.softmax(self.encoder_s(obs["x"]), dim=-1)
        # q(v | x)
        v_probs = F.softmax(self.encoder_v(obs["x"]), dim=-1)
        # q(r | x)
        r_probs = F.softmax(self.encoder_r(obs["x"]), dim=-1)
        # q(f | x, v) 
        f_probs = F.softmax(self.encoder_f(obs["x"], y=obs["view"]), dim=-1)
        # q(a | x, f, v) 
        a_loc, _ = self.encoder_a(obs["x"], y=torch.cat([obs["finding"],obs["view"]],dim=1)).chunk(2, dim=-1)

        return {
            "sex": s_probs,
            "race": r_probs,
            "view": v_probs,
            "finding": f_probs,
            "age": a_loc,
        }

    def svi_model(self, **obs) -> None:
        with pyro.plate("observations", obs["x"].shape[0]):
            pyro.condition(self.model, data=obs)()

    def guide_pass(self, **obs) -> None:
        pass


class ChestPGM_PE(BasePGM):
    def __init__(self, args: Hparams, temperature: float = 1.0):

        super().__init__()

        self.variables = {
            "race": "categorical",       # 3 categories - Asian, White, Black
            "sex": "categorical",        # 2 categories - Male, Female
            "view": "categorical",       # 2 categories - PA, AP
            "pe_finding": "categorical", # 2 categories - NoFinding, PE
            "age": "continuous",         # 0 - 100
        }

        # parent: {children}
        self.graph = {
            "race": {"x"},    
            "sex": {"x"},     
            "view": {"x"},    
            "pe_finding": {"x","view"}, 
            "age": {"x","view","pe_finding"},      
        }

        race_shape = 3
        sex_shape = 2
        view_shape = 2
        pe_finding_shape = 2
        age_shape = 1

        # log space for sex, race, finding, and view
        self.sex_logits = nn.Parameter(np.log(1 / sex_shape) * torch.ones(1, sex_shape))
        self.race_logits = nn.Parameter(np.log(1 / race_shape) * torch.ones(1, race_shape))

        # define base distributions: 
        # age (continuous),
        self.a_base_loc = nn.Parameter(torch.zeros(age_shape))
        self.a_base_scale = nn.Parameter(torch.zeros(age_shape))  
        # finding-pe (categorical with condition - Gumbel),
        self.register_buffer("fp_base_loc", torch.zeros(pe_finding_shape))
        self.register_buffer("fp_base_scale", torch.ones(pe_finding_shape))
        # view (categorical with condition - Gumbel)
        self.register_buffer("v_base_loc", torch.zeros(view_shape))
        self.register_buffer("v_base_scale", torch.ones(view_shape))

        # age spline flow
        self.age_flow_components = T.ComposeTransformModule([T.Spline(age_shape, count_bins=5, order="linear")])
        self.age_flow = T.ComposeTransform([self.age_flow_components,])

        # age -> pe_finding : pe_finding - gumble - loc and scale
        pe_finding_net = DenseNN(age_shape, [8,16], param_dims=[pe_finding_shape, pe_finding_shape], nonlinearity=nn.ReLU())
        self.pe_finding_flow = ConditionalAffineTransform(pe_finding_net, event_dim=1)

        # [age,finding] -> view : view - gumble - loc and scale
        # adding pe_finding + 1 as gumble softmax centered trick adds one dummy dimension
        view_net = DenseNN(age_shape + pe_finding_shape + 1, [8,16], param_dims=[view_shape, view_shape], nonlinearity=nn.ReLU())        
        self.view_flow = ConditionalAffineTransform(view_net, event_dim=1)

        # initialize softmax transform with temperature
        self.softmax_transform = SoftmaxCentered(temperature)

        ####################################
        # define anticausal predictors

        shared_model = ResNet(CustomBlock, layers=[2, 2, 2, 2], widths=[64, 128, 256, 512], norm_layer=lambda c: nn.GroupNorm(min(32, c // 4), c),)
        shared_model.conv1 = nn.Conv2d(args.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False,)
        kwargs = {"in_shape": (args.input_channels, *(args.input_res,) * 2), "base_model": shared_model,}

        # q(s | x) ~ OneHotCategorical(f(x))
        self.encoder_s = ResNet18(num_outputs=sex_shape, **kwargs)

        # q(v | x) ~ OneHotCategorical(f(x)) 
        # additional shape for view due to softmax-centered trick
        self.encoder_v = ResNet18(num_outputs=view_shape+1, **kwargs)

        # q(r | x) ~ OneHotCategorical(f(x))
        self.encoder_r = ResNet18(num_outputs=race_shape, **kwargs)

        # q(fp | x, v) ~ OneHotCategorical(f(x)) 
        # q(fc | x, v) ~ OneHotCategorical(f(x)) 
        # additional shape for finding due to softmax-centered trick
        # additional shape for view due to softmax-centered trick
        self.encoder_fp = ResNet18(num_outputs=pe_finding_shape+1, context_dim=view_shape+1, **kwargs)

        # q(a | x, fp, fc, v) ~ Normal(mu(x), sigma(x))
        # additional shape for finding due to softmax-centered trick
        # additional shape for view due to softmax-centered trick
        self.encoder_a = ResNet18(num_outputs=age_shape+age_shape, context_dim=view_shape+1+pe_finding_shape+1, **kwargs)

    def model(self) -> Dict[str, Tensor]:

        pyro.module("ChestPGM_PE", self)

        # p(r)
        pr = dist.OneHotCategorical(logits=self.race_logits).to_event(1)
        race = pyro.sample("race", pr)

        # p(s)
        ps = dist.OneHotCategorical(logits=self.sex_logits).to_event(1)
        sex = pyro.sample("sex", ps)

        # p(a)
        pa_base = dist.Normal(self.a_base_loc, F.softplus(self.a_base_scale, beta=np.log(2))).to_event(1)
        pa = dist.TransformedDistribution(pa_base, self.age_flow)
        age = pyro.sample("age", pa)

        # p(fp|a)
        pfp_a_base = dist.Gumbel(self.fp_base_loc, self.fp_base_scale).to_event(1)
        pfp_a = ConditionalTransformedDistribution(pfp_a_base, [self.pe_finding_flow, self.softmax_transform],).condition(age)
        pe_finding = pyro.sample("pe_finding", pfp_a)

        # p(v|fp,fc,a)
        pv_fpa_base = dist.Gumbel(self.v_base_loc, self.v_base_scale).to_event(1)
        pv_fpa = ConditionalTransformedDistribution(pv_fpa_base, [self.view_flow, self.softmax_transform],).condition(torch.cat([age,pe_finding],dim=1)) #because softmax centered adds a dummy dimension
        view = pyro.sample("view", pv_fpa)

        return {
            "sex": sex,
            "race": race,
            "age": age,
            "pe_finding": pe_finding,
            "view": view,
        }

    def model_anticausal(self, **obs) -> None:
        # assumes all variables are observed, train classfiers
        pyro.module("ChestPGM_PE", self)

        with pyro.plate("observations", obs["x"].shape[0]):

            # q(s | x)
            s_probs = F.softmax(self.encoder_s(obs["x"]), dim=-1)
            qs_x = dist.OneHotCategorical(probs=s_probs).to_event(1)
            pyro.sample("sex_aux", qs_x, obs=obs["sex"])

            # q(v | x)
            v_probs = F.softmax(self.encoder_v(obs["x"]), dim=-1)
            qv_x = dist.OneHotCategorical(probs=v_probs).to_event(1)
            pyro.sample("view_aux", qv_x, obs=obs["view"])

            # q(r | x)
            r_probs = F.softmax(self.encoder_r(obs["x"]), dim=-1)
            qr_x = dist.OneHotCategorical(probs=r_probs).to_event(1)
            pyro.sample("race_aux", qr_x, obs=obs["race"])

            # q(fp | x, v)
            fp_probs = F.softmax(self.encoder_fp(obs["x"], y=obs["view"]), dim=-1) 
            qfp_xv = dist.OneHotCategorical(probs=fp_probs).to_event(1)
            pyro.sample("pe_finding_aux", qfp_xv, obs=obs["pe_finding"])

            # q(a | x, fp, v)
            a_loc, a_logscale = self.encoder_a(obs["x"], y=torch.cat([obs["pe_finding"],obs["view"]],dim=1)).chunk(2, dim=-1) 
            qa_xfpv = dist.Normal(a_loc, F.softplus(a_logscale, beta=np.log(2))).to_event(1)
            pyro.sample("age_aux", qa_xfpv, obs=obs["age"])

    def predict(self, **obs) -> Dict[str, Tensor]:
        # q(s | x)
        s_probs = F.softmax(self.encoder_s(obs["x"]), dim=-1)
        # q(v | x)
        v_probs = F.softmax(self.encoder_v(obs["x"]), dim=-1)
        # q(r | x)
        r_probs = F.softmax(self.encoder_r(obs["x"]), dim=-1)
        # q(fp | x, v) 
        fp_probs = F.softmax(self.encoder_fp(obs["x"], y=obs["view"]), dim=-1)
        # q(a | x, fp, v) 
        a_loc, _ = self.encoder_a(obs["x"], y=torch.cat([obs["pe_finding"],obs["view"]],dim=1)).chunk(2, dim=-1)

        return {
            "sex": s_probs,
            "race": r_probs,
            "view": v_probs,
            "pe_finding": fp_probs,
            "age": a_loc,
        }

    def svi_model(self, **obs) -> None:
        with pyro.plate("observations", obs["x"].shape[0]):
            pyro.condition(self.model, data=obs)()

    def guide_pass(self, **obs) -> None:
        pass

