from __future__ import print_function

import torch
import argparse
import random
import os
import numpy as np
import json
import yaml

from trainer.unet2d_trainer import unet2d_trainer

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

    parser.add_argument('--epochs', help='number of epochs. Default: 240', type=int, nargs='?', default=240, const=240)
    parser.add_argument('--resumetraining', action="store_true", help="resumer training flag. If 1, latest epoch is loaded and training is restarted from there.")

    parser.add_argument('--lr', help='Learning Rate for AdamW Optimizer. Default:0.001', type=float, nargs='?', default=0.001, const=0.001)
    parser.add_argument('--lrdecay', help='Learning Rate Decay for Adam Optimizer. Default:0.995', type=float, nargs='?', default=0.995, const=0.995)
    parser.add_argument('--b1', help='B1 for Adam Optimizer. Default:0.9', type=float, nargs='?', default=0.9, const=0.9)
    parser.add_argument('--b2', help='B2 for Adam Optimizer. Default:0.999', type=float, nargs='?', default=0.999, const=0.999)
    parser.add_argument('--wd', help='Weight Decay for Adam Optimizer. Default:0.0001', type=float, nargs='?', default=0.0001, const=0.0001)
    parser.add_argument('--batchsize', help='batch size. Default:12', type=int, nargs='?', default=12, const=12)

    parser.add_argument('--outputdir', help='path to the output directory. Default:/data2/rmehta3/Experiments/', type=str, nargs='?', default='/data2/rmehta3/Experiments/', const='/data2/rmehta3/Experiments/')
    parser.add_argument('--experimentname', help='name of the experiment. A directory with this name is created within outputdir, and all outputs are stored there. Default: SSN_2D_LIDC', type=str, nargs='?', default='SSN_2D_LIDC', const='SSN_2D_LIDC')
    parser.add_argument('--workers', help='number of data loading workers. Default:8', type=int, nargs='?', default=8, const=8)

    parser.add_argument('--dataset', help="name of the dataset. options:[mimic,mimic_pe,mimic_cm]. Default:mimic", type=str, nargs='?', default='mimic', const='mimic')

    parser.add_argument('--inputpath', help='path to the input directory. Default:/data2/rmehta3/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_files_preprocessed/', type=str, nargs='?', default='/data2/rmehta3/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_files_preprocessed/', const='/data2/rmehta3/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_files_preprocessed/')
    parser.add_argument('--labelpath', help='path to the label directory. Default:/data2/rmehta3/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_segmentation_preprocessed/', type=str, nargs='?', default='/data2/rmehta3/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_segmentation_preprocessed/', const='/data2/rmehta3/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_segmentation_preprocessed/')
    parser.add_argument('--csvpath', help='path to the csv directory. Default:/data2/rmehta3/datasets/chest_xray/', type=str, nargs='?', default='/data2/rmehta3/datasets/chest_xray', const='/data2/rmehta3/datasets/chest_xray/')
    parser.add_argument('--traincsv', help='train csv file name. Default: train_pe_cm.csv', type=str, nargs='?', default='train_pe_cm.csv', const='train_pe_cm.csv')
    parser.add_argument('--validcsv', help='valid csv file name. Default: valid_pe_cm.csv', type=str, nargs='?', default='valid_pe_cm.csv', const='valid_pe_cm.csv')
    parser.add_argument('--testcsv', help='test csv file name. Default: test_pe_cm.csv', type=str, nargs='?', default='test_pe_cm.csv', const='test_pe_cm.csv')
    parser.add_argument('--rcathreshold', help='RCA threshold. Default:0.8', type=float, nargs='?', default=0.8, const=0.8)
    parser.add_argument('--scalerange', help="if data augmentation is used, then range of scaling parameter. Default:0.2", type=float, nargs='?', default=0.2, const=0.2)
    parser.add_argument('--rotationdegree', help="if data augmentation is used, then range of rotation degree. Default:10", type=float, nargs='?', default=10.0, const=10.0)
    parser.add_argument('--augment', action="store_true", help="if this flag is used then data augmentation is used during training.")

    parser.add_argument('--imagefreq', help='Image Are visualized after every imagefreq epochs. Default:5', type=int, nargs='?', default=5, const=5)

    args = parser.parse_args()
    argparse_dict = vars(args)

    os.makedirs(os.path.join(args.outputdir, args.experimentname,'config'), exist_ok=True)
    with open(os.path.join(args.outputdir, args.experimentname,'config','config.yaml'), "w") as outfile:
        yaml.dump(argparse_dict, outfile)

    set_seed(args.seed)

    # device
    if args.gpu is None:
        device = torch.device('cpu')
    else:
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda", args.gpu)


    trainer = unet2d_trainer(   
        dataset = args.dataset,              
        nlayers = args.nlayers,
        nscales = args.nscales,
        init_kern = args.initkern,
        dropout_rate = args.dropoutrate,
        norm = args.norm,
        activ = args.activation,
        pad_type = args.padtype,
        ds_type = args.dstype,
        us_type = args.ustype,
        merge_type = args.mergetype,
        inp_chan = args.inpchan,
        out_ch = args.classes,
        epochs = args.epochs,
        resume_training = args.resumetraining,
        device = device,
        lr = args.lr,
        b1 = args.b1,
        b2 = args.b2,
        wd = args.wd,
        lr_decay = args.lrdecay,
        batch_size = args.batchsize,
        output_dir = args.outputdir,
        experiment_name = args.experimentname,
        workers = args.workers,
        input_path = args.inputpath,
        label_path = args.labelpath,
        csv_path = args.csvpath,
        train_csv = args.traincsv,
        valid_csv = args.validcsv,
        test_csv = args.testcsv,
        rca_threshold= args.rcathreshold, 
        scale_range = args.scalerange,
        rotation_degree = args.rotationdegree,
        augment = args.augment,
        image_freq = args.imagefreq,
    )

    trainer.main_worker()

if __name__ == "__main__":
    main()