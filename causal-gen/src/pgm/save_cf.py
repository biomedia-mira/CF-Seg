import argparse
import copy
import os
import random
import send2trash
import sys
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import torch
from dscm import DSCM, vae_preprocess, padchest_var_shape, padchest_var_values
from flow_pgm import ChestPGM, ChestPGM_CM, ChestPGM_PE, ChestPGM_LD, ChestPGM_O, ChestPGM_AT, ChestPGM_ED, ChestPGM_NP, ChestPGM_CO
from tqdm import tqdm
from train_pgm import preprocess, setup_dataloaders

sys.path.append("..")
from hps import Hparams
from utils import EMA, seed_all, generate_combinations
from vae import HVAE



def setup_save_directories(args: Hparams) -> str:
    parents_folder = "_".join([k[0] for k in args.interven_variables])
    save_dir = os.path.join(args.save_path, parents_folder, args.save_folder, args.save_set)
    if os.path.isdir(save_dir):
        if (input(f"\nSave directory '{save_dir}' already exists, overwrite? [y/N]: ") == "y"):
            if input(f"Send '{save_dir}', to Trash? [y/N]: ") == "y":
                send2trash.send2trash(save_dir)
                print("Done.\n")
            else:
                exit()
        else:
            if (input(f"\nResume training with save directory '{save_dir}'? [y/N]: ") == "y"):
                pass
            else:
                exit()
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def get_title_mimic(do: Dict, dicom_ids: List, age_group: torch.tensor):

   dicom_ids_updated = []
   for j, id in enumerate(dicom_ids):
      msg = id
      for i, (k, v) in enumerate(do.items()):
         if k == "sex":
            sex_categories = ["male", "female"]  # 0,1
            vv = sex_categories[int(torch.argmax(v[j], dim=-1))]
            kk = "sex"
         elif k == "view":
            view_categories = ["AP", "PA"]  # 0,1
            vv = view_categories[int(torch.argmax(v[j], dim=-1))]
            kk = "view"
         elif k == "race":
            race_categories = ["white", "asian", "black"]  # 0,1,2
            vv = race_categories[int(torch.argmax(v[j], dim=-1))]
            kk = "race"
         elif k == "finding":
            finding_categories = ["NoFinding", "PE", "CM", "PE&CM"]  # 0,1,2,3
            vv = finding_categories[int(torch.argmax(v[j], dim=-1))]
            kk = "finding"
         elif k == "pe_finding":
            finding_categories = ["NoFinding", "PE"]  # 0,1
            vv = finding_categories[int(torch.argmax(v[j], dim=-1))]
            kk = "finding"
         elif k == "age":
            vv = str(int(age_group[j].item()))
            kk = "age_group"
         else:
            raise NotImplementedError
         msg += "_"+ kk + "_" + vv
      dicom_ids_updated.append(msg)
   return dicom_ids_updated


if __name__ == "__main__":

   parser = argparse.ArgumentParser()
   # generic
   parser.add_argument("--dataset", help="Dataset name.", type=str, default="mimic")
   parser.add_argument("--seed", help="random seed.", type=int, default=7)
   parser.add_argument("--bs", help="batch size.", type=int, default=64)
   parser.add_argument("--deterministic", help="toggle cudNN determinism.", action="store_true", default=False,)
   parser.add_argument("--load_path", help="Full Path to load checkpoint.", type=str, default="")
   parser.add_argument("--pgm_predictor_path", help="path to load pgm_predictor checkpoint.", type=str, default="../../checkpoints/sup_pgm/checkpoint.pt",)
   parser.add_argument("--vae_path", help="path to load vae checkpoint.", type=str, default="../../checkpoints/from_server/m_b_v_s/ukbb192_beta5_dgauss_b33/checkpoint.pt",)
   parser.add_argument("--save_path",help="Full path where we want to save counterfactuals", type=str, default="")
   parser.add_argument("--save_folder",help="Folder name to save counterfactuals", type=str, default="")
   parser.add_argument("--save_set", help="save counterfactuals for save_set. Options: train, test, valid. Defauld: train", type=str, default="train")
   # dataloader
   parser.add_argument("--inputpath", help="Data directory to load form.", type=str, default="")
   parser.add_argument("--labelpath", help="Label (segmentation) directory to load form.", type=str, default="")
   parser.add_argument("--csvpath", help="path to directory where csv files are stored.", type=str, default="")
   parser.add_argument("--train_csv", help="csv file name for train set.", type=str, default="train_pe_cm.csv")
   parser.add_argument("--test_csv", help="csv file name for test set.", type=str, default="test_pe_cm.csv")
   parser.add_argument("--valid_csv", help="csv file name for valid set.", type=str, default="valid_pe_cm.csv")
   parser.add_argument("--rca_threshold", help="rca threshold for mimic dataframe", type=float, default=0.8)
   # counterfactual
   parser.add_argument("--cf_particles", help="num counterfactual samples.", type=int, default=1)
   parser.add_argument("--interven_variables", help="Variables to Interven on.", nargs="+",default=["view", "age", "race", "sex", "finding"],)
   parser.add_argument("--temparature", help="temparature for vae.", type=float, default=1.0)
   parser.add_argument("--u_temparature", help="temparature for counterfacutual.", type=float, default=1.0)
   parser.add_argument("--input_channels", help="number of input channels", type=int, default=1)
   parser.add_argument("--pred_race", help="if true then race is predicted using anti-causal. Useful for ASRD. Default: False.", action="store_true", default=False,)

   args = parser.parse_known_args()[0]

   seed_all(args.seed, args.deterministic)

   if args.dataset not in ["mimic","mimic_pe"]:
      raise NotImplementedError("dataset should be either mimic or mimic_pe")

   save_dir = setup_save_directories(args)

   #######
   print(f"\nLoading pgm_predictor checkpoint: {args.pgm_predictor_path}")
   pgm_predictor_checkpoint = torch.load(args.pgm_predictor_path)

   print(f"\nLoading VAE checkpoint: {args.vae_path}")
   vae_checkpoint = torch.load(args.vae_path)

   print(f"\nLoading DSCM checkpoint: {args.load_path}")
   ckpt = torch.load(args.load_path)

   #######    
   ckpt_args = {k: v for k, v in ckpt["hparams"].items() if k not in vars(args).keys()} # donot take args which are defined here
   vars(args).update(ckpt_args)
   pgm_predictor_args = {k: v for k, v in pgm_predictor_checkpoint["hparams"].items() if k not in vars(args).keys()} # donot take args which are defined here
   vars(args).update(pgm_predictor_args)
   vae_args = {k: v for k, v in vae_checkpoint["hparams"].items() if k not in vars(args).keys()} # donot take args which are defined here
   vars(args).update(vae_args)

   ###### define PGM #####

   if args.dataset == "mimic":
      pgm_predictor = ChestPGM(args).cuda()
      print(f"\nDefined MIMIC PGM")
   elif args.dataset == "mimic_pe":
      pgm_predictor = ChestPGM_PE(args).cuda()
      print(f"\nDefined MIMIC_PE PGM")
   else:
      NotImplementedError("dataset should be either mimic or mimic_pe")

   ###### define VAE ######
   vae = HVAE(args).cuda()

   ###### define DSCM #####
   model = DSCM(args, pgm_predictor, pgm_predictor, vae)
   model.cuda()
   model.load_state_dict(ckpt["ema_model_state_dict"])
   model.vae.eval()
   model.pgm.eval()
   model.predictor.eval()

   ##### setup dataloaders #####
   dataloaders = setup_dataloaders(args, augment=False, drop_last=False)
   # select based on save_set
   dataloader = dataloaders[args.save_set]

   # we will use this for interating through each combination of variable values
   # use only the intervened variables
   cfs_variable_values = {}
   for v in args.interven_variables:
      if args.dataset in ["mimic","mimic_pe"]:
         cfs_variable_values[v] = padchest_var_values[v]
      else:
         raise NotImplementedError("dataset should be either mimic or mimic_pe")

   # iterate throught each combination of intervened variables
   # ex. cfs_variable_values = {"sex": [0,1], "view": [0,1]}
   # output: [{"sex":0, "view":0},{"sex":0, "view":1},{"sex":1, "view":0},{"sex":1, "view":1}]
   # print(cfs_variable_values)
   iter_dicts = generate_combinations(cfs_variable_values)
   # print(iter_dicts)

   for dct in iter_dicts:
      dir_name = ("_").join(f"{k}_{v}" for k,v in dct.items())
      os.makedirs(os.path.join(save_dir,dir_name), exist_ok=True)

   with torch.no_grad():

      # loader = tqdm(enumerate(dataloader), total=len(dataloader), miniters=len(dataloader) // 100, mininterval=5,)

      for i, batch in enumerate(dataloader):

         bs = batch["x"].shape[0]
         batch = preprocess(batch)

         pa = {k: v for k, v in batch.items() if k not in ["x", "y", "dicom_id"]}

         ### we will need to predict race here, as PadChest doesn't have race given
         # print(batch.keys())
         pred = model.predictor.predict(**batch)
         if args.pred_race:
            pa["race"] = pred["race"].clone().cuda()

         _pa = vae_preprocess(args, {k: v.clone() for k, v in pa.items()})

         # forward vae and get abducted noise
         zs = model.vae.abduct(batch["x"], parents=_pa, t=args.temparature)  # z ~ q(z|x,pa)

         if model.vae.cond_prior:
            zs = [zs[j]["z"] for j in range(len(zs))]

         rec_loc, rec_scale = model.vae.forward_latents(zs, parents=_pa)


         # now iter through each combination of intervened variables and generate counterfactuals for them
         iter_dict_tqdm = tqdm(enumerate(iter_dicts), total=len(iter_dicts))

         for _, iter_dict in iter_dict_tqdm:
            # print(f"iter_dict: {iter_dict}")

            dir_name = "_".join(f"{k}_{v}" for k,v in iter_dict.items())   
            save_dir_iter_dict = os.path.join(save_dir,dir_name)     

            iter_dict_tqdm.set_description(f"batch:{i:05d} of {len(dataloader)} || "
                                 + f", ".join(f"{k}:{v}" for k,v in iter_dict.items())
                                 + " ||")

            # generate do variables for each image in the batch 
            do = {}
            for v in model.pgm.variables.keys():
               if v in args.interven_variables: # take values based on inter_var_values
                  # print(f"inter_var: {v}")
                  if model.pgm.variables[v] == "categorical":
                     do[v] = torch.zeros_like(batch[v]).cuda() # generate zeros
                     do[v][:,iter_dict[v]] = 1  # convert into one hot based on value
                  elif model.pgm.variables[v] == "continuous":
                     if v == "age":
                        # if the age range is same as the original age then just take the original age
                        do[v] = torch.zeros_like(batch[v]).cuda() # generate zeros
                        for b in range(len(batch[v])):
                           if iter_dict[v] == (((batch[v][b]+1)*50)//20):
                              do[v][b] = batch[v][b].clone().cuda()
                           else:
                              do[v][b] = ((((torch.randint(iter_dict[v]*20,(iter_dict[v]+1)*20,(1,))) / 100) * 2) - 1).cuda()
                        age_group = iter_dict[v]*torch.ones_like(batch[v]).cuda()
               else: # non-intervened
                  if ((v == "race") and (args.pred_race)):   
                     do[v] = pred["race"].clone().cuda()
                  else:
                     do[v] = batch[v].clone().cuda() # directly copy original values

                  if v == "age":
                     age_group = ((batch[v] + 1)*50)//20
                     age_group = age_group.cuda()

            image_id = batch["dicom_id"]

            # get parents and counterfactual parents
            _cf_pa = vae_preprocess(args, {k: v.clone() for k, v in do.items()})

            if args.cf_particles > 1:  # for calculating counterfactual uncertainty
               cfs = {"x": torch.zeros_like(batch["x"])}
               cfs.update({"x2": torch.zeros_like(batch["x"])})

            for _ in range(args.cf_particles):

               # get recon and cf loc and scale
               cf_loc, cf_scale = model.vae.forward_latents(zs, parents=_cf_pa)

               # apply temp to cf_scale
               cf_scale = cf_scale * args.u_temparature

               # get recon error and use to rescale cf_scale
               u = (batch["x"] - rec_loc) / rec_scale.clamp(min=1e-12)

               # get cfs and rescale intensity range
               cf_x = torch.clamp(cf_loc + cf_scale * u, min=-1, max=1) # [-1,1]
               cf_x = (cf_x + 1) / 2.0 # [0,1]

               # calculate cf statistics
               if args.cf_particles > 1:
                  cfs["x"] += cf_x
                  with torch.no_grad():
                     cfs["x2"] += cf_x**2
               else:
                  cfs = {"x": cf_x}
            
            # Var[X] = E[X^2] - E[X]^2
            if args.cf_particles > 1:
               with torch.no_grad():
                  var_cf_x = (cfs["x2"] - cfs["x"] ** 2 / args.cf_particles) / args.cf_particles
                  cfs.pop("x2", None)
               cfs["x"] = cfs["x"] / args.cf_particles
            else:
               var_cf_x = None

            # convert to numpy
            cfs["x"] = cfs["x"].cpu().detach().numpy()
            var_cf_x = var_cf_x.cup().detach().numpy() if args.cf_particles > 1 else None   
            cfs["x"] = np.squeeze(cfs["x"])
            var_cf_x = np.squeeze(var_cf_x)

            # save images
            for b in range(bs):
               img = Image.fromarray(np.uint8(cfs["x"][b,...] * 255), "L")
               img.save(f"{save_dir_iter_dict}/{image_id[b]}_cf.png")

            # save variance maps
            if args.cf_particles > 1:
               for b in range(b):
                  mean_var = np.mean(var_cf_x[b,...])
                  img = Image.fromarray(np.uint8(var_cf_x[b,...] * 255), "L")
                  img.save(f"{save_dir_iter_dict}/{image_id[b]}_cf_var.png")
      
