import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ExponentialLR 

import time
import statistics
import os
import matplotlib
matplotlib.use('Agg')
import json
import random

import numpy as np
import pandas as pd

from math import log10
from os.path import join
from tqdm import tqdm
from shutil import copyfile

from models.unet2D_module import Unet2D
from datagenerator.datagenerator import CheXMaskDataGenerator
from custom_callbacks.logger import Logger
from custom_callbacks.loss_plotter import LossPlotter
from custom_callbacks.visualizer import show_samples
from utils.metrics import dice, batch_dice
from utils.checkpoint_utils import load_checkpoint, save_checkpoint, get_model


chexmask_var = {
    "sex":{"0":"male","1":"female"},
    "race":{"0":"white","1":"asian","2":"black"},
    "vp":{"0":"AP", "1":"PA"},
    "disease":{"0":"NoFinding", "1":"PleuralEffusion", "2":"Cardigomagely", "3":"PleuralEffusion&Cardigomagely"}
}




######################################################################################################################
### U-Net 2D
######################################################################################################################

class unet2d_trainer:
    """
    Main Segmentation Handler -- UNet2D network                     
    """

    def __init__(self,
                 dataset: str = "mimic",
                 nlayers: int = 2,
                 nscales: int = 4,
                 init_kern: int = 32,
                 dropout_rate: float = 0.5,
                 norm: str = 'in',
                 activ: str ='lrelu',
                 pad_type: str ='zero',
                 ds_type: str = "max",
                 us_type: str = "transpose",
                 merge_type: str = "concate",
                 inp_chan: int = 1,
                 out_ch: int = 1,
                 epochs: int = 500,
                 resume_training: bool = False,
                 device: torch.device = torch.device("cuda"),
                 lr: float = 0.001,
                 b1: float = 0.9,
                 b2: float = 0.999,
                 wd: float = 0.0001,
                 lr_decay: float = 0.995,
                 batch_size: int = 12,
                 output_dir: str = "/vol/biomedic3/rmehta3/Experiments/SSN_2D/",
                 experiment_name: str = "MIMIC_UNet2D",
                 workers: int = 8,
                 input_path: str = "/vol/biomedic3/rmehta3/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_files_preprocessed/",
                 label_path: str = "/vol/biomedic3/rmehta3/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_segmentation_preprocessed/",
                 csv_path: str = "/vol/biomedic3/rmehta3/datasets/chest_xray/",
                 train_csv: str = "train_pe_only.csv",   
                 test_csv: str = "test_pe_only.csv",   
                 valid_csv: str = "valid_pe_only.csv",
                 rca_threshold: float = 0.8,   
                 scale_range: float = 0.2,
                 rotation_degree: float = 10.0,
                 augment: bool = True,
                 image_freq: int = 5,
                ):

        """
        Trainer for UNet 2D
        """

        self.device = device
        self.batch_size = batch_size
        self.inp_chan = inp_chan
        self.total_epochs = epochs
        self.initial_epoch = 0
        self.out_ch = out_ch
        self.image_freq = image_freq
        self.dataset = dataset


        # make all necessary paths
        self.log_path = os.path.join(output_dir, experiment_name, 'log')
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(os.path.join(self.log_path, 'weights'), exist_ok=True)
        os.makedirs(os.path.join(self.log_path, 'images'), exist_ok=True)

        self.model = Unet2D(n_layers=nlayers,
                            n_scales=nscales,
                            init_ker=init_kern,
                            inp_chan=inp_chan,
                            out_chan=out_ch,
                            dropout_rate=dropout_rate,
                            norm=norm,
                            activ=activ,
                            pad_type=pad_type,
                            ds_type=ds_type,
                            us_type=us_type,
                            merge_type=merge_type,
                            last_layer="default")


        print(self.model)
        self.model = self.model.to(device)
        print(f"===> Model Defined.")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, betas=(b1,b2), weight_decay=wd)
        print(f"===> AdamW Optimzer Defined. LR: {lr}, (B1,B2): ({b1,b2}, weight_decay: {wd})")

        if dataset in ["mimic", "mimic_pe"]:
            train_DG = CheXMaskDataGenerator(inputpath=input_path, labelpath=label_path, csvpath=os.path.join(csv_path,train_csv), rca_threshold=rca_threshold, augment=augment, scale_range=scale_range, rotation_degree=rotation_degree, subset="train")   
            valid_DG = CheXMaskDataGenerator(inputpath=input_path, labelpath=label_path, csvpath=os.path.join(csv_path,valid_csv), rca_threshold=rca_threshold, augment=False, subset="valid")   
            test_DG = CheXMaskDataGenerator(inputpath=input_path, labelpath=label_path, csvpath=os.path.join(csv_path,test_csv), rca_threshold=rca_threshold, augment=False, subset="test")   
        else:
            raise Exception("Invalid dataset")

        self.train_data_loader = DataLoader(train_DG, shuffle=True, batch_size=batch_size, num_workers=workers, drop_last=True)
        self.valid_data_loader = DataLoader(valid_DG, shuffle=False, batch_size=batch_size, num_workers=workers, drop_last=True)
        self.test_data_loader = DataLoader(test_DG, shuffle=False, batch_size=batch_size, num_workers=workers, drop_last=True)

        print(f"===> DataLoaders defined.")
        print(f"===> Train. Batches: {len(self.train_data_loader)}, Batch_size: {batch_size}, Samples: {len(train_DG)}")
        print(f"===> Valid. Batches: {len(self.valid_data_loader)}, Batch_size: {batch_size}, Samples: {len(valid_DG)}")
        print(f"===> Test. Batches: {len(self.test_data_loader)}, Batch_size: {batch_size}, Samples: {len(test_DG)}")

        # define criterion for backprop
        self.crit = nn.BCEWithLogitsLoss()

        self.test_crit = nn.BCEWithLogitsLoss(reduction="none")
        print("===> Loss Defined Initialized")

        # define all metric and loss to monitor
        self.metric_monitor = {}
        self.metric_monitor["loss"] = {'mode':'min', 'best_state':np.inf, 'epoch':0}
        self.metric_monitor["d_mean"] = {'mode':'max', 'best_state':0.0, 'epoch':0}
        for i in range(out_ch):
            self.metric_monitor["d_"+str(i)] = {'mode':'max', 'best_state':0.0, 'epoch':0}

        self.metric_names = list(self.metric_monitor.keys())

        # setup our callbacks
        self.logger = Logger(log_path=self.log_path, log_name="training.log", metric_names=self.metric_names)
        self.LP = LossPlotter(log_path=self.log_path, log_name="training.log", metric_names=self.metric_names)
        print("===> Logger and LossPlotter Initialized")

        # check if we want to resume training from the last saved epoch
        if resume_training:
            self.model, self.optimizer, self.initial_epoch, self.metric_monitor = load_checkpoint(save_type="latest_epoch", model=self.model, optimizer=self.optimizer, log_path=self.log_path)

        self.current_epoch = self.initial_epoch

        # initialize LR scheduler
        # self.scheduler = ExponentialLR(self.optimizer, gamma=LR_decay, last_epoch=self.initial_epoch - 1)
        # print("===> Learning Rate Scheduler Initialized")


    def train_epoch(self):

        """
        main training loop -- iterates through each batch of training DataLoader, calculates loss, updates model weights
                                using optimizer, calculates loss and dice

        returns:
                metric -- average metric and loss across training dataset
        """

        self.model.train()

        metric = np.zeros(len(self.metric_names))

        for iteration, batch in enumerate(tqdm(self.train_data_loader)):

            batch_metrics = []

            self.optimizer.zero_grad()

            # inp, target = batch['input'].to(self.device), batch['output'].type('torch.LongTensor').to(self.device)
            inp, target = batch['input'].to(self.device), batch['output'].type('torch.FloatTensor').to(self.device)

            output = self.model(inp)

            # we have total 3 binary segmentation - left lung, right lung, and heart
            loss = 0

            for i in range(self.out_ch):
                loss += self.crit(output[:,i,...], target[:,i,...])

            loss.backward()
            batch_metrics += [loss.item()]

            self.optimizer.step()

            # calculate dice score for each structure of interest
            output = F.sigmoid(output)
            outp = torch.zeros_like(output, device=output.device)
            outp[output>=0.5] = 1.0

            dsc = []

            for i in range(self.out_ch):
                dsc.append(batch_dice(outp[:,i,...],target[:,i,...],2))

            dsc = torch.stack(dsc) # (out_ch,B,2)
            dsc = dsc[:,:,1] # we ignore background classes (out_ch,B)
            dsc = dsc.transpose(0,1)  # (B,out_ch)
            dsc = dsc.mean(dim=0) # (out_ch)

            batch_metrics += [torch.mean(dsc).item()] # add mean dsc
            batch_metrics += list(dsc.cpu().numpy())

            metric += np.array(batch_metrics)

        # self.scheduler.step()

        return metric / len(self.train_data_loader)


    def validate_epoch(self):

        """
        main validation loop -- iterates through each batch of validation DataLoader, calculates loss and dice

        returns:
                metric -- average metric and loss across validation dataset
        """

        self.model.eval()

        metric = np.zeros(len(self.metric_names))

        with torch.no_grad():

            for iteration, batch in enumerate(tqdm(self.valid_data_loader)):

                batch_metrics = []

                # inp, target = batch['input'].to(self.device), batch['output'].type('torch.LongTensor').to(self.device)
                inp, target = batch['input'].to(self.device), batch['output'].type('torch.FloatTensor').to(self.device)

                batch_size = inp.shape[0]

                output = self.model(inp)

                # we have total 3 binary segmentation - left lung, right lung, and heart
                loss = 0
                for i in range(self.out_ch):
                    loss += self.crit(output[:,i,...], target[:,i,...])

                batch_metrics += [loss.item()]

                # calculate dice score for each structure of interest
                output = F.sigmoid(output)
                outp = torch.zeros_like(output, device=output.device)
                outp[output>=0.5] = 1.0

                dsc = []

                for i in range(self.out_ch):
                    dsc.append(batch_dice(outp[:,i,...],target[:,i,...],2))

                dsc = torch.stack(dsc) # (out_ch,B,2)
                dsc = dsc[:,:,1] # we ignore background classes (out_ch,B)
                dsc = dsc.transpose(0,1)  # (B,out_ch)
                dsc = dsc.mean(dim=0) # (out_ch)

                batch_metrics += [torch.mean(dsc).item()] # add mean dsc
                batch_metrics += list(dsc.cpu().numpy())

                metric += np.array(batch_metrics)

            if self.current_epoch%self.image_freq == 0:
                # at the end of save samples from the last batch to visualize the output samples
                images = torch.cat((inp, target, outp), dim=1) # (B,3+OC+OC,H,W)
                images = images[:,2::,...]
                # hack as show_images uses subplots and they don't work with single example and two indices
                if batch_size == 1:
                    images = images.repeat_interleave(2,dim=0) 
                row_headers = [f"Example-{i}" for i in range(images.shape[0])]
                column_headers = ["Input"]+[f"Target-{i}" for i in range(target.shape[1])]+[f"Pred-{i}" for i in range(outp.shape[1])]
                show_samples(images.cpu().numpy(), row_headers, column_headers, self.current_epoch, os.path.join(self.log_path, 'images'))        

        return metric / len(self.valid_data_loader)


    def test_epoch(self):

        """
        main test loop -- iterates through each batch of test DataLoader, calculates loss and dice

        returns:
                metric -- pandas dataframe containing 
        """

        self.model.eval()

        metric = np.zeros(len(self.metric_names))

        dct = {"age":[],
            "sex":[],
            "vp":[], 
            "race":[],
            "disease":[],
            "dicom_id":[]} 

        for nm in self.metric_names:
            dct[nm] = []

        with torch.no_grad():

            for iteration, batch in enumerate(tqdm(self.test_data_loader)):

                # inp, target = batch['input'].to(self.device), batch['output'].type('torch.LongTensor').to(self.device)
                inp, target = batch['input'].to(self.device), batch['output'].type('torch.FloatTensor').to(self.device)

                if self.dataset in ["mimic", "mimic_pe"]:
                    dct["age"] += list(batch["age"].cpu().numpy())
                    dct["sex"] += list(batch["sex"].cpu().numpy())
                    dct["race"] += list(batch["race"].cpu().numpy())
                    dct["vp"] += list(batch["vp"].cpu().numpy())
                    dct["disease"] += list(batch["disease"].cpu().numpy())
                    dct["dicom_id"] += batch["dicom_id"]

                output = self.model(inp)

                # we have total 3 binary segmentation - left lung, right lung, and heart
                loss = np.zeros(inp.shape[0])
                for i in range(self.out_ch):
                    ind_loss = self.test_crit(output[:,i,...], target[:,i,...]) # (B,H,W)
                    loss += ind_loss.mean((1,2)).cpu().numpy() # (B,)
                dct["loss"] += list(loss/self.out_ch)

                # calculate dice score for each structure of interest
                output = F.sigmoid(output)
                outp = torch.zeros_like(output, device=output.device)
                outp[output>=0.5] = 1.0

                dsc = []

                for i in range(self.out_ch):
                    dsc.append(batch_dice(outp[:,i,...],target[:,i,...],2))

                dsc = torch.stack(dsc) # (out_ch,B,2)
                dsc = dsc[:,:,1] # we ignore background classes (out_ch,B)
                dsc = dsc.transpose(0,1)  # (B,out_ch)

                for i in range(self.out_ch):
                    dct["d_"+str(i)] += list(dsc[:,i].cpu().numpy())

                dct["d_mean"] += list(dsc.mean(dim=-1).cpu().numpy())

        return pd.DataFrame(dct)



    def main_worker(self):

        """
        main worker which run through all training, validation, and testing functions

        return:
                test_metric: metrics on testing dataset
        """

        print(f"===> Starting Model Training at Epoch: {self.initial_epoch}")
        print(f"===> Total Trainable Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

        for epch in range(self.initial_epoch, self.total_epochs):

            start = time.time()

            print("\n\n")
            print(f"Epoch:{epch}")

            self.current_epoch = epch

            train_metric = self.train_epoch()
            print(f"===> Training   Epoch: {epch:03d} " + ", ".join([f"{key}: {value:.4f}" for value, key in zip(train_metric, self.metric_monitor.keys())]))

            valid_metric = self.validate_epoch()
            print(f"===> Validation Epoch: {epch:03d} " + ", ".join([f"{key}: {value:.4f}" for value, key in zip(valid_metric, self.metric_monitor.keys())]))

            self.logger.to_csv(np.concatenate((train_metric, valid_metric)), epch)
            print("===> Logged All Metrics")

            self.LP.plotter()
            print("===> Plotted All Metrics")

            # check if any the current model is best for any metrics, if yes, save that model weight
            for i, key in enumerate(self.metric_monitor.keys()):
                best_state = self.metric_monitor[key]['best_state']
                update = valid_metric[i]>best_state if self.metric_monitor[key]['mode'] == 'max' else valid_metric[i]<best_state
                if update:
                    self.metric_monitor[key]['best_state']=valid_metric[i]
                    self.metric_monitor[key]['epoch']=epch
                    save_checkpoint(self.model,self.optimizer,epch,self.metric_monitor,self.log_path,save_type="best_"+key)

            # save latest model weight    
            save_checkpoint(self.model,self.optimizer,epch,self.metric_monitor,self.log_path,save_type="latest_epoch")

            end = time.time()

            print(f"===> Epoch:{epch:03d} Completed in {(end-start):.2f} seconds")

        print("\n\n")
        print(f"===> Done Training for Total {self.total_epochs} Epochs")
        print("\n\n\n\n")

        for i, key in enumerate(self.metric_monitor.keys()): 
            self.model, self.optimizer, best_epoch, self.metric_monitor = load_checkpoint(save_type="best_"+key, model=self.model, optimizer=self.optimizer, log_path=self.log_path)
            print(f"===> Loaded Best Model weights based on {key} with value {self.metric_monitor[key]['best_state']:.4f} at epoch {best_epoch}")
            test_df = self.test_epoch()
            print("===> Test Results")
            print(f"===> Mean Dice Overall: {test_df['d_mean'].mean():.4f}")

            if self.dataset in ["mimic","mimic_pe"]:
                for u in test_df['sex'].unique():
                    print(f"===> Mean Dice for {chexmask_var['sex'][str(u)]}: {test_df[test_df['sex']==u]['d_mean'].mean():.4f}")
                for u in test_df['race'].unique():
                    print(f"===> Mean Dice for {chexmask_var['race'][str(u)]}: {test_df[test_df['race']==u]['d_mean'].mean():.4f}")
                for u in test_df['vp'].unique():
                    print(f"===> Mean Dice for {chexmask_var['vp'][str(u)]}: {test_df[test_df['vp']==u]['d_mean'].mean():.4f}")
                for u in test_df['disease'].unique():
                    print(f"===> Mean Dice for {chexmask_var['disease'][str(u)]}: {test_df[test_df['disease']==u]['d_mean'].mean():.4f}")
            else:
                print("===> No Subgroup Analysis for this dataset")
            
            print("\n")
            test_df.to_csv(os.path.join(self.log_path,f"test_results_for_best_{key}.csv"),index=False,header=True)

        print("\n\n")
        print("===> Saved All Test Results in a csv file")