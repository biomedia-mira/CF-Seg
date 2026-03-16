import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import einops


def IOU_cont(preds: torch.tensor,
             targets: torch.tensor,
             num_classes: int = 2,
              conv_targets_to_oh: bool = True,
              conv_preds_to_oh: bool = False):
    """
    Args:
        - preds: predicted tensor. Shape - B,H,W,.. or B,C,H,W,...
        - targets: target tensor. Shape - B,H,W,... or B,C,H,W,...
        - num_classes: total number of classes C. Shape - 1
        - conv_target_to_oh: if True target is converted to one-hot else it is not.
        - conv_target_to_oh: if True target is converted to one-hot else it is not.
    """
    if conv_preds_to_oh:
        # in this case expected shape of preds is B,H,W,...
        # otherwise it is B,C,H,W,....    
        # convert into one hot encoding
        class_idx = einops.rearrange(torch.arange(num_classes), 'C -> 1 C').to(preds.device) # (1,C)
        preds_oh = (preds[...,None] == class_idx).int() # (B,H,W,...,C)
        preds_oh = einops.rearrange(preds_oh, 'B ... C -> B C ...') # (B,C,H,W,...)
    else:
        preds_oh = preds

    if conv_targets_to_oh:
        # in this case expected shape of targets is B,H,W,...
        # otherwise it is B,C,H,W,....    
        # convert into one hot encoding
        class_idx = einops.rearrange(torch.arange(num_classes), 'C -> 1 C').to(targets.device) # (1,C)
        targets_oh = (targets[...,None] == class_idx).int() # (B,H,W,...,C)
        targets_oh = einops.rearrange(targets_oh, 'B ... C -> B C ...') # (B,C,H,W,...)
    else:
        targets_oh = targets

    assert preds_oh.shape == targets_oh.shape, f"shape mismatch between preds_oh and targets_oh."
    assert preds_oh.shape[1] == num_classes, f"preds_oh num_channels (C:{preds_oh.shape[1]}) is not same num_classes:{num_classes}."
    assert targets_oh.shape[1] == num_classes, f"targets_oh num_channels (C:{targets_oh.shape[1]}) is not same num_classes:{num_classes}."

    # calculate TP, FP, and FN
    TP = preds_oh*targets_oh # (B,C,H,W,...)
    TP = einops.reduce(TP, 'B C ... -> B C', 'sum') # (B,C)

    FP = preds_oh*(1.0-targets_oh) # (B,C,H,W,...)
    FP = einops.reduce(FP, 'B C ... -> B C', 'sum') # (B,C)

    FN = (1.0-preds_oh)*(targets_oh) # (B,C,H,W,...)
    FN = einops.reduce(FN, 'B C ... -> B C', 'sum') # (B,C)

    # calculate iou
    iou = (TP) / (TP+FP+FN) # (B,C)

    # is 0/0 then convert into 1
    iou[torch.isnan(iou)] = 1.0 # (B,C)

    # mean across all classes except 0 as it is background class in medical imaging context
    iou = torch.mean(iou[:,1:], dim=1) # (B)

    return iou

def DICE_cont(preds: torch.tensor,
              targets: torch.tensor,
              num_classes: int = 2,
              conv_targets_to_oh: bool = True,
              conv_preds_to_oh: bool = False):
    """
    Args:
        - preds: predicted tensor. Shape - B,H,W,.. or B,C,H,W,...
        - targets: target tensor. Shape - B,H,W,... or B,C,H,W,...
        - num_classes: total number of classes C. Shape - 1
        - conv_target_to_oh: if True target is converted to one-hot else it is not.
        - conv_target_to_oh: if True target is converted to one-hot else it is not.
    """
    if conv_preds_to_oh:
        # in this case expected shape of preds is B,H,W,...
        # otherwise it is B,C,H,W,....    
        # convert into one hot encoding
        class_idx = einops.rearrange(torch.arange(num_classes), 'C -> 1 C').to(preds.device) # (1,C)
        preds_oh = (preds[...,None] == class_idx).int() # (B,H,W,...,C)
        preds_oh = einops.rearrange(preds_oh, 'B ... C -> B C ...') # (B,C,H,W,...)
    else:
        preds_oh = preds

    if conv_targets_to_oh:
        # in this case expected shape of targets is B,H,W,...
        # otherwise it is B,C,H,W,....    
        # convert into one hot encoding
        class_idx = einops.rearrange(torch.arange(num_classes), 'C -> 1 C').to(targets.device) # (1,C)
        targets_oh = (targets[...,None] == class_idx).int() # (B,H,W,...,C)
        targets_oh = einops.rearrange(targets_oh, 'B ... C -> B C ...') # (B,C,H,W,...)
    else:
        targets_oh = targets

    assert preds_oh.shape == targets_oh.shape, f"shape mismatch between preds_oh and targets_oh."
    assert preds_oh.shape[1] == num_classes, f"preds_oh num_channels (C:{preds_oh.shape[1]}) is not same num_classes:{num_classes}."
    assert targets_oh.shape[1] == num_classes, f"targets_oh num_channels (C:{targets_oh.shape[1]}) is not same num_classes:{num_classes}."

    # calculate TP, FP, and FN
    TP = preds_oh*targets_oh # (B,C,H,W,...)
    TP = einops.reduce(TP, 'B C ... -> B C', 'sum') # (B,C)

    FP = preds_oh*(1.0-targets_oh) # (B,C,H,W,...)
    FP = einops.reduce(FP, 'B C ... -> B C', 'sum') # (B,C)

    FN = (1.0-preds_oh)*(targets_oh) # (B,C,H,W,...)
    FN = einops.reduce(FN, 'B C ... -> B C', 'sum') # (B,C)

    # calculate dice
    dice = (2*TP) / ((2*TP)+FP+FN) # (B,C)

    # is 0/0 then convert into 1
    dice[torch.isnan(dice)] = 1.0 # (B,C)

    # mean across all classes except 0 as it is background class in medical imaging context
    dice = torch.mean(dice[:,1:], dim=1) # (B)

    return dice
