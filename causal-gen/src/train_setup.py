import logging
import os
from typing import Any, Dict, Tuple

import send2trash
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from datasets import mimic
from hps import Hparams
from utils import linear_warmup, seed_worker

mimic_finding_name = {
                   "mimic":"finding",
                   "mimic_pe":"pe_finding",
                   }

def setup_dataloaders(args: Hparams, augment=True, drop_last=True, weightedsampler=False) -> Dict[str, DataLoader]:
    if args.dataset in ["mimic","mimic_pe"]:
        datasets = mimic(args, augment)
    else:
        NotImplementedError

    kwargs = {"batch_size": args.bs, "num_workers": 8, "pin_memory": True, "worker_init_fn": seed_worker,}
    dataloaders = {}
    if weightedsampler:
        sample_labels_oh = datasets["train"].samples[mimic_finding_name[args.dataset]]
        sample_labels = sample_labels_oh.argmax(dim=1)
        class_counts = torch.bincount(sample_labels)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[sample_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        dataloaders["train"] = DataLoader(datasets["train"], sampler=sampler, **kwargs)
    else:
        dataloaders["train"] = DataLoader(datasets["train"], shuffle=True, drop_last=drop_last, **kwargs)
   
    dataloaders["valid"] = DataLoader(datasets["valid"], shuffle=False, drop_last=drop_last, **kwargs)
    dataloaders["test"] = DataLoader(datasets["test"], shuffle=False, drop_last=drop_last, **kwargs)
   
    return dataloaders


def setup_optimizer(args: Hparams, model: nn.Module) -> Tuple[torch.optim.Optimizer, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=args.betas)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_warmup(args.lr_warmup_steps))
    return optimizer, scheduler

def setup_directories(args: Hparams, ckpt_dir: str = "../checkpoints") -> str:
    parents_folder = "_".join([k[0] for k in args.parents_x])
    save_dir = os.path.join(ckpt_dir, parents_folder, args.exp_name)
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


def setup_tensorboard(args: Hparams, model: nn.Module) -> SummaryWriter:
    """Setup metric summary writer."""
    writer = SummaryWriter(args.save_dir)

    hparams = {}
    for k, v in vars(args).items():
        if isinstance(v, list) or isinstance(v, torch.device):
            hparams[k] = str(v)
        elif isinstance(v, torch.Tensor):
            hparams[k] = v.item()
        else:
            hparams[k] = v

    writer.add_hparams(hparams, {"hparams": 0}, run_name=os.path.abspath(args.save_dir))

    if "vae" in type(model).__name__.lower():
        z_str = []
        if hasattr(model.decoder, "blocks"):
            for i, block in enumerate(model.decoder.blocks):
                if block.stochastic:
                    z_str.append(f"z{i}_{block.res}x{block.res}")
        else:
            z_str = ["z0_" + str(args.z_dim)]

        writer.add_custom_scalars(
            {
                "nelbo": {"nelbo": ["Multiline", ["nelbo/train", "nelbo/valid"]]},
                "nll": {"kl": ["Multiline", ["nll/train", "nll/valid"]]},
                "kl": {"kl": ["Multiline", ["kl/train", "kl/valid"]]}
            }
        )
    return writer


def setup_logging(args: Hparams) -> logging.Logger:
    # reset root logger
    [logging.root.removeHandler(h) for h in logging.root.handlers[:]]
    # info logger for saving command line outputs during training
    logging.basicConfig(
        handlers=[logging.FileHandler(os.path.join(args.save_dir, "trainlog.txt")), logging.StreamHandler(),],
        format="%(asctime)s, %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(args.exp_name)  # name the logger
    return logger
