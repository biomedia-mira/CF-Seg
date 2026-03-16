import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import jaccard_index

def dice(pred: torch.tensor,
         target: torch.tensor,
         num_classes: int = 2):

    """
    Function to calculate dice between pred and target

    Args:
        pred : prediction torch tensor (H,W,...)
        target: target torch tensor (H,W,...)
        num_classes: number of total possible classes C

    returns:
        dsc : torch.tensor (C,)
    """
    assert pred.shape == target.shape

    dsc = []

    for i in range(num_classes):

        # convert into binary with forground == i    
        pred_bin = torch.zeros_like(pred, device=pred.device)
        target_bin = torch.zeros_like(pred, device=pred.device)
        
        pred_bin[pred==i] = 1.0
        target_bin[target==i] = 1.0

        # calculate intersection and union
        intersection = torch.sum(pred_bin*target_bin)
        union = torch.sum(pred_bin) + torch.sum(target_bin)

        if union == 0.0: # if union is zero then there is no forground class in either target or pred so dc is 1.0
            dc = 1.0
        else:
            dc = (2*intersection) / (union)

        dsc.append(dc)
    
    return torch.tensor(dsc)



def batch_dice(preds: torch.tensor,
               targets: torch.tensor,
               num_classes: int = 2):
    """
    Function to calculate dice between pred and target for each sample

    Args:
        pred : prediction torch tensor (B,H,W,...)
        target: target torch tensor (B,H,W,...)
        num_classes: number of total possible classes C

    returns:
        dsc : torch.tensor (B,C)

    """

    assert preds.shape == targets.shape

    bs = preds.shape[0]

    b_dsc = []
    for i in range(bs):
        b_dsc.append(dice(preds[i,...],targets[i,...],num_classes=num_classes))

    return torch.stack(b_dsc) # (B,C)

