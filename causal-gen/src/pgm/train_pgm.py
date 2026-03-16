import argparse
import copy
import os
import sys
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pyro
import torch
from layers import TraceStorage_ELBO
from sklearn.metrics import roc_auc_score, recall_score
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils_pgm import update_stats

sys.path.append("..")
from datasets import mimic
from hps import Hparams
from train_setup import setup_directories, setup_logging, setup_tensorboard, setup_dataloaders
from utils import EMA, seed_all, seed_worker

from flow_pgm import ChestPGM, ChestPGM_CM, ChestPGM_PE, ChestPGM_LD, ChestPGM_O, ChestPGM_AT, ChestPGM_ED, ChestPGM_NP, ChestPGM_CO

mimic_finding_name = {
                   "mimic":"finding",
                   "mimic_pe":"pe_finding",
                   }



def preprocess(batch: Dict[str, Tensor], dataset: str = "ukbb") -> Dict[str, Tensor]:

    if "x" in batch.keys(): 
        batch["x"] = batch["x"].float().cuda()  # [-1,1]

    # for all other variables except x and dicom_id
    not_x = [k for k in batch.keys() if k not in ["x","y","dicom_id"]]
    for k in not_x:
        batch[k] = batch[k].float().cuda()
        if len(batch[k].shape) < 2:
            batch[k] = batch[k].unsqueeze(-1)
    return batch


def sup_epoch(args: Hparams, 
              model: nn.Module,
              ema: Optional[nn.Module], 
              dataloader: Dict[str, DataLoader], 
              elbo_fn: TraceStorage_ELBO, 
              elbo_fn_anticausal: TraceStorage_ELBO, 
              optimizer: Optional[torch.optim.Optimizer] = None,
              is_train: bool = True,
             ) -> Dict[str, Any]:
    
    stats = {"loss": 0, "loss_sup": 0, "loss_aux": 0, "n": 0}  # sample counter
    
    loader = tqdm(enumerate(dataloader), total=len(dataloader), miniters=len(dataloader) // 100, mininterval=5,)

    model.train(is_train)

    for i, batch in loader:

        bs = batch["x"].shape[0]
        batch = preprocess(batch, args.dataset)

        with torch.set_grad_enabled(is_train):
            loss_anticausal = elbo_fn_anticausal.differentiable_loss(model.model_anticausal, model.guide_pass, **batch) / bs
            loss_sup = elbo_fn.differentiable_loss(model.svi_model, model.guide_pass, **batch) / bs
            loss = loss_anticausal + loss_sup

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 200)
            optimizer.step()
            ema.update()

        stats["loss"] += loss.item() * bs
        stats["loss_sup"] += loss_sup.item() * bs
        stats["loss_aux"] += loss_anticausal.item() * bs
        stats["n"] += bs
        stats = update_stats(stats, elbo_fn)
        loader.set_description(f' => {("train" if is_train else "eval")} | '
                             + f", ".join(f'{k}: {v / stats["n"]:.4f}' for k, v in stats.items() if k != "n")
                             + f", grad_norm: {grad_norm:.3f}" if is_train else "",  # refresh=False
                              )
    return {k: v / stats["n"] for k, v in stats.items() if k != "n"}


@torch.no_grad()
def eval_epoch(args: Hparams, model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
    # "caution: this can consume lots of memory if dataset is large"
    model.eval()
    preds = {k: [] for k in model.variables.keys()}
    targets = {k: [] for k in model.variables.keys()}

    for batch in tqdm(dataloader):
        for k in targets.keys():
            targets[k].extend(copy.deepcopy(batch[k]))
        # predict
        batch = preprocess(batch, args.dataset)
        out = model.predict(**batch)

        for k, v in out.items():
            preds[k].extend(v)

    for k, v in preds.items():
        preds[k] = torch.stack(v).squeeze().cpu()
        targets[k] = torch.stack(targets[k]).squeeze()
   
    stats = {}
    for k in model.variables.keys():
        if k == "age":
            preds_k = (preds[k] + 1) * 50  # unormalize
            targets_k = (targets[k] + 1) * 50  # unormalize
            stats[k + "_mae"] = (targets_k - preds_k).abs().mean().item()
        elif k in ["race","sex"]:
            num_corrects = (targets[k].argmax(-1) == preds[k].argmax(-1)).sum()
            stats[k + "_acc"] = num_corrects.item() / targets[k].shape[0]
            stats[k + "_rocauc"] = roc_auc_score(targets[k].numpy(), preds[k].numpy(), multi_class="ovr", average="macro",)
        elif k in ["finding","view","pe_finding"]:
            num_corrects = (targets[k].argmax(-1) == preds[k].argmax(-1)).sum()
            stats[k + "_acc"] = num_corrects.item() / targets[k].shape[0]
            stats[k + "_b_acc"] = recall_score(targets[k].argmax(-1).numpy(), preds[k].argmax(-1).numpy(), average="macro")
            stats[k + "_rocauc"] = roc_auc_score(targets[k].numpy()[:,:-1], preds[k].numpy()[:,:-1], multi_class="ovr", average="macro",)
        else:
            NotImplementedError
    
    return stats


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # generic
    parser.add_argument("--exp_name", help="Experiment name.", type=str, default="")
    parser.add_argument("--dataset", help="Dataset name.", type=str, default="mimic")
    parser.add_argument("--seed", help="Set random seed.", type=int, default=7)
    parser.add_argument("--deterministic", help="Toggle cudNN determinism.", action="store_true", default=False,)
    # dataloader
    parser.add_argument("--inputpath", help="Data directory to load form.", type=str, default="")
    parser.add_argument("--labelpath", help="Label (segmentation) directory to load form.", type=str, default="")
    parser.add_argument("--csvpath", help="path to directory where csv files are stored.", type=str, default="")
    parser.add_argument("--train_csv", help="csv file name for train set.", type=str, default="train_pe_cm.csv")
    parser.add_argument("--test_csv", help="csv file name for test set.", type=str, default="test_pe_cm.csv")
    parser.add_argument("--valid_csv", help="csv file name for valid set.", type=str, default="valid_pe_cm.csv")
    parser.add_argument("--rca_threshold", help="rca threshold for mimic dataframe.", type=float, default=0.8)
    parser.add_argument("--weightedsampler", help="if True, weightedRandomSampler is used.", action="store_true", default=False)
    # testing
    parser.add_argument("--testing", help="Test model.", action="store_true", default=False)
    parser.add_argument("--load_path", help="Path to load checkpoint.", type=str, default="")
    # training
    parser.add_argument("--epochs", help="Number of training epochs.", type=int, default=1000)
    parser.add_argument("--bs", help="Batch size.", type=int, default=32)
    parser.add_argument("--lr", help="Learning rate.", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps", help="lr warmup steps.", type=int, default=1)
    parser.add_argument("--wd", help="Weight decay penalty.", type=float, default=0.1)
    parser.add_argument("--input_res", help="Input image crop resolution.", type=int, default=256)
    parser.add_argument("--input_channels", help="Input image num channels.", type=int, default=1)
    parser.add_argument("--scale_range", help="Augmentation: Scale in the range of 1+-scale_range.", type=float, default=0.1)
    parser.add_argument("--rotation_degree", help="Augmentation: Rotation (degree) in the range of +-rotation_degree.", type=float, default=10.0)
    parser.add_argument("--eval_freq", help="Num epochs per eval.", type=int, default=1)
    # model
    parser.add_argument("--parents_x", help="Parents of x to load.", nargs="+", default=["sex", "view", "finding", "age", "race"])
    parser.add_argument("--add_dummy_dim", help="add dummy dimension for softmax_centered trick in PGM", action="store_true", default=False)
    parser.add_argument("--dummy_var", help="Variable which requires added dummy dimensions in PGM.", nargs="+", default=["view", "finding"])
    parser.add_argument("--std_fixed", help="Fix aux dist std value (0 is off).", type=float, default=0)
    args = parser.parse_known_args()[0]

    seed_all(args.seed, args.deterministic)

    # Load data
    dataloaders = setup_dataloaders(args, weightedsampler=args.weightedsampler)

    # Init model
    pyro.clear_param_store()
    if args.dataset == "mimic":
        model = ChestPGM(args)
    elif args.dataset == "mimic_pe":
        model = ChestPGM_PE(args)
    else:
        NotImplementedError

    ema = EMA(model, beta=0.999)
    model.cuda()
    ema.cuda()
    print(model.variables.keys())

    # Init loss & optimizer
    elbo_fn = TraceStorage_ELBO(num_particles=2)
    elbo_fn_anticausal = TraceStorage_ELBO(num_particles=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if not args.testing:

        # Train model
        args.save_dir = setup_directories(args, ckpt_dir="../../checkpoints")
        writer = setup_tensorboard(args, model)
        logger = setup_logging(args)

        for k in sorted(vars(args)):
            logger.info(f"--{k}={vars(args)[k]}")

        args.best_loss = float("inf")
        args.best_b_acc_finding = 0.0


        for epoch in range(args.epochs):

            logger.info(f"Epoch {epoch+1}:")

            stats = sup_epoch(args, model, ema, dataloaders["train"], elbo_fn, elbo_fn_anticausal, optimizer, is_train=True,)

            if epoch % args.eval_freq == 0:

                valid_stats = sup_epoch(args, ema.ema_model, None, dataloaders["valid"], elbo_fn, elbo_fn_anticausal, None, is_train=False,)

                steps = (epoch + 1) * len(dataloaders["train"])

                logger.info(f'loss            | train: {stats["loss"]:.4f} - valid: {valid_stats["loss"]:.4f} - steps: {steps}')
                logger.info(f'loss-sup        | train: {stats["loss_sup"]:.4f} - valid: {valid_stats["loss_sup"]:.4f} - steps: {steps}')
                logger.info(f'loss-anticausal | train: {stats["loss_aux"]:.4f} - valid: {valid_stats["loss_aux"]:.4f} - steps: {steps}')

                for k, v in stats.items():
                    writer.add_scalar("train/" + k, v, steps)
                    writer.add_scalar("valid/" + k, valid_stats[k], steps)

                writer.add_custom_scalars({"elbo": {"elbo": ["Multiline", ["elbo/train", "elbo/valid"]]}})
                writer.add_scalar("elbo/train", stats["loss"], steps)
                writer.add_scalar("elbo/valid", valid_stats["loss"], steps)

                metrics = eval_epoch(args, ema.ema_model, dataloaders["valid"])
                logger.info("valid | " + " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

            if valid_stats["loss"] < args.best_loss:
                args.best_loss = valid_stats["loss"]
                ckpt_path = os.path.join(args.save_dir, "checkpoint.pt")
                torch.save({"epoch": epoch + 1, 
                            "step": steps, 
                            "best_loss": args.best_loss, 
                            "model_state_dict": model.state_dict(),
                            "ema_model_state_dict": ema.ema_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "hparams": vars(args),},
                            ckpt_path,)
                logger.info(f"Model saved: {ckpt_path} for Epoch {epoch}")

            if metrics[f"{mimic_finding_name[args.dataset]}_b_acc"] > args.best_b_acc_finding:
                args.best_b_acc_finding = metrics[f"{mimic_finding_name[args.dataset]}_b_acc"]
                ckpt_path = os.path.join(args.save_dir, "checkpoint_b_acc.pt")
                torch.save({"epoch": epoch + 1, 
                            "step": steps, 
                            "best_loss": args.best_b_acc_finding, 
                            "model_state_dict": model.state_dict(),
                            "ema_model_state_dict": ema.ema_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "hparams": vars(args),},
                            ckpt_path,)
                logger.info(f"Model saved: {ckpt_path} for Epoch {epoch}")

        # load the best model based on validation set loss for testing
        ckpt = torch.load(ckpt_path)

    else:
        # update hparams if loading checkpoint
        if args.load_path:
            if os.path.isfile(args.load_path):
                print(f"\nLoading checkpoint: {args.load_path}")
                ckpt = torch.load(args.load_path)
                ckpt_args = {k: v for k, v in ckpt["hparams"].items() if k != "load_path"}
                if args.data_dir is not None:
                    ckpt_args["data_dir"] = args.data_dir
                if args.testing:
                    ckpt_args["testing"] = args.testing
                vars(args).update(ckpt_args)
            else:
                print(f"Checkpoint not found at: {args.load_path}")
                NotImplementedError

    # test model
    logger.info(f"Testing Model for best val loss at {ckpt['epoch']} epoch")
    model.load_state_dict(ckpt["model_state_dict"])
    ema.ema_model.load_state_dict(ckpt["ema_model_state_dict"])
    print("Evaluating test set:\n")

    # eval PGM
    stats = sup_epoch(args, ema.ema_model, None, dataloaders["test"], elbo_fn, elbo_fn_anticausal, None, is_train=False,)

    # eval AUX-AntiCausal
    stats = eval_epoch(args, ema.ema_model, dataloaders["test"])
    logger.info("test | " + " - ".join(f"{k}: {v:.4f}" for k, v in stats.items()))
