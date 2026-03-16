from __future__ import print_function

import torch
import argparse
import random
import os
import numpy as np
import json
import yaml
import pandas as pd

from trainer.unet2d_trainer import unet2d_trainer

from models.unet2D_module import Unet2D
from datagenerator.datagenerator import CheXMaskDataGenerator
from custom_callbacks.logger import Logger
from custom_callbacks.loss_plotter import LossPlotter
from custom_callbacks.visualizer import show_samples
from utils.metrics import dice, batch_dice
from utils.checkpoint_utils import load_checkpoint, save_checkpoint, get_model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ExponentialLR 

from tqdm import tqdm

mimic_var_categories = {"sex": ["male", "female"],         
                        "view": ["AP","PA"],         
                        "finding": ["NoFinding", "PE", "CM", "PE&CM"],  
                        "pe_finding": ["NoFinding", "PE"],   
                        "cm_finding": ["NoFinding", "CM"],   
                        "age": [0,1,2,3,4],    
                        "race": ["white", "asian", "black"],
                        }      


############################################################################################

def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

####################################################################################

def main():

    parser = argparse.ArgumentParser(description="Main handler for training", usage="python ./train.py -g 0")
    parser.add_argument('--gpu', help='GPU number', type=int, nargs='?', default=None, const=0)
    parser.add_argument('--seed', help='Random Seed', type=int, nargs='?', default=42, const=42)

    parser.add_argument('--nlayers', help='Number of layers at each scale in Unet. Default:2', type=int, nargs='?', default=2, const=2)
    parser.add_argument('--nscales', help='Number of scales in Unet. Default:4', type=int, nargs='?', default=4, const=4)
    parser.add_argument('--initkern', help='Number of initial Kernels in Unet. Default:24', type=int, nargs='?', default=24, const=24)
    parser.add_argument('--dropoutrate', help='Dropout Rate. Default:0.5', type=float, nargs='?', default=0.5, const=0.5)
    parser.add_argument('--norm', help='Normalization Type. Default:in', type=str, nargs='?', default='in', const='in')
    parser.add_argument('--activation', help='Activation Type. Default:lrelu', type=str, nargs='?', default='lrelu', const='lrelu')
    parser.add_argument('--padtype', help='Convolution Padding Type. Default:zero', type=str, nargs='?', default='zero', const='zero')
    parser.add_argument('--dstype', help='Downsampling Type in U-Net. Default:max', type=str, nargs='?', default='max', const='max')
    parser.add_argument('--ustype', help='Upsampling Type in U-Net. Default:transpose', type=str, nargs='?', default='transpose', const='transpose')
    parser.add_argument('--mergetype', help='Merging Type of same scales of encoder and decoder in U-Net. Default:concate', type=str, nargs='?', default='concate', const='concate')

    parser.add_argument('--inpchan', help='number of input channerls. Default:1', type=int, nargs='?', default=1, const=1)
    parser.add_argument('--classes', help='number of output classes. Default:1', type=int, nargs='?', default=1, const=1)

    parser.add_argument('--lr', help='Learning Rate for AdamW Optimizer. Default:0.001', type=float, nargs='?', default=0.001, const=0.001)
    parser.add_argument('--lrdecay', help='Learning Rate Decay for Adam Optimizer. Default:0.995', type=float, nargs='?', default=0.995, const=0.995)
    parser.add_argument('--b1', help='B1 for Adam Optimizer. Default:0.9', type=float, nargs='?', default=0.9, const=0.9)
    parser.add_argument('--b2', help='B2 for Adam Optimizer. Default:0.999', type=float, nargs='?', default=0.999, const=0.999)
    parser.add_argument('--wd', help='Weight Decay for Adam Optimizer. Default:0.0001', type=float, nargs='?', default=0.0001, const=0.0001)
    parser.add_argument('--batchsize', help='batch size. Default:12', type=int, nargs='?', default=12, const=12)

    parser.add_argument('--outputdir', help='path to the output directory. Default:/data2/rmehta3/Experiments/', type=str, nargs='?', default='/data2/rmehta3/Experiments/', const='/data2/rmehta3/Experiments/')
    parser.add_argument('--experimentname', help='name of the experiment. A directory with this name is created within outputdir, and all outputs are stored there. Default: SSN_2D_LIDC', type=str, nargs='?', default='SSN_2D_LIDC', const='SSN_2D_LIDC')
    parser.add_argument('--workers', help='number of data loading workers. Default:8', type=int, nargs='?', default=8, const=8)

    parser.add_argument('--loadcheckpointtype', help='checkpoint type to load, Ex. best_d_mean', type=str, nargs='?', default='best_d_mean', const='best_d_mean')

    parser.add_argument('--dataset', help="name of the dataset. options:[mimic,mimic_pe,mimic_cm]. Default:mimic", type=str, nargs='?', default='mimic', const='mimic')

    parser.add_argument('--inputpath', help='path to the input directory. Default:/data2/rmehta3/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_files_preprocessed/', type=str, nargs='?', default='/data2/rmehta3/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_files_preprocessed/', const='/data2/rmehta3/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_files_preprocessed/')
    parser.add_argument('--labelpath', help='path to the label directory. Default:/data2/rmehta3/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_segmentation_preprocessed/', type=str, nargs='?', default='/data2/rmehta3/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_segmentation_preprocessed/', const='/data2/rmehta3/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_segmentation_preprocessed/')
    parser.add_argument('--csvpath', help='path to the csv directory. Default:/data2/rmehta3/datasets/chest_xray/', type=str, nargs='?', default='/data2/rmehta3/datasets/chest_xray', const='/data2/rmehta3/datasets/chest_xray/')
    parser.add_argument('--testcsv', help='test csv file name. Default: test_pe_cm.csv', type=str, nargs='?', default='test_pe_cm.csv', const='test_pe_cm.csv')
 

    args = parser.parse_args()

    set_seed(args.seed)

    # device
    if args.gpu is None:
        device = torch.device('cpu')
    else:
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda", args.gpu)


    model = Unet2D(n_layers=args.nlayers,
                    n_scales=args.nscales,
                    init_ker=args.initkern,
                    inp_chan=args.inpchan,
                    out_chan=args.classes,
                    dropout_rate=args.dropoutrate,
                    norm=args.norm,
                    activ=args.activation,
                    pad_type=args.padtype,
                    ds_type=args.dstype,
                    us_type=args.ustype,
                    merge_type=args.mergetype,
                    last_layer="default")

    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, 
                            betas=(args.b1,args.b2),
                            weight_decay=args.wd)

    test_crit = nn.BCEWithLogitsLoss(reduction="none")

    log_path = os.path.join(args.outputdir, args.experimentname, 'log') 

    model, optimizer, initial_epoch, metric_monitor = load_checkpoint(save_type=args.loadcheckpointtype, model=model, optimizer=optimizer, log_path=log_path)

    metric_names = list(metric_monitor.keys())


    test_DG = CheXMaskDataGenerator(inputpath=args.inputpath,
                                    labelpath=args.labelpath,
                                    csvpath=os.path.join(args.csvpath,args.testcsv),
                                    rca_threshold=args.rcathreshold,
                                    augment=False,
                                    subset="test")

    test_data_loader = DataLoader(test_DG, shuffle=False, batch_size=args.batchsize, num_workers=args.workers, drop_last=True)


    ######### iterate through test dataloader #########

    model.eval()        

    metric = np.zeros(len(metric_names))

    dct = {"age":[],
        "sex":[],
        "vp":[], 
        "race":[],
        "disease":[],
        "dicom_id":[]} 

    for nm in metric_names:
        dct[nm] = []


    with torch.no_grad():

        for iteration, batch in enumerate(tqdm(test_data_loader)):

            inp, target = batch['input'].to(device), batch['output'].type('torch.FloatTensor').to(device)

            dct["age"] += list(batch["age"].cpu().numpy())
            dct["sex"] += list(batch["sex"].cpu().numpy())
            dct["race"] += list(batch["race"].cpu().numpy())
            dct["vp"] += list(batch["vp"].cpu().numpy())
            dct["disease"] += list(batch["disease"].cpu().numpy())
            dct["dicom_id"] += batch["dicom_id"]

            output = model(inp)

            # we have total 3 binary segmentation - left lung, right lung, and heart
            loss = np.zeros(inp.shape[0])
            for i in range(args.classes):
                ind_loss = test_crit(output[:,i,...], target[:,i,...]) # (B,H,W)
                loss += ind_loss.mean((1,2)).cpu().numpy() # (B,)
            dct["loss"] += list(loss/args.classes)

            # calculate dice score for each structure of interest
            output = F.sigmoid(output)
            outp = torch.zeros_like(output, device=output.device)
            outp[output>=0.5] = 1.0

            dsc = []

            for i in range(args.classes):
                dsc.append(batch_dice(outp[:,i,...],target[:,i,...],2))

            dsc = torch.stack(dsc) # (out_ch,B,2)
            dsc = dsc[:,:,1] # we ignore background classes (out_ch,B)
            dsc = dsc.transpose(0,1)  # (B,out_ch)

            for i in range(args.classes):
                dct["d_"+str(i)] += list(dsc[:,i].cpu().numpy())

            dct["d_mean"] += list(dsc.mean(dim=-1).cpu().numpy())

    test_df = pd.DataFrame(dct)

    test_df.to_csv(os.path.join(log_path,f"test_results_for_{args.loadcheckpointtype}.csv"),index=False,header=True)

if __name__ == "__main__":
    main()