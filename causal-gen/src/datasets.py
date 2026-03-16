import gzip
import os
import random
import struct
from typing import Dict, List, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as TF
from torchvision import tv_tensors
from torchvision.transforms import v2

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from hps import Hparams

import cv2

# from pgm.dscm import mimic_var_shape, mimic_var_values


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



mimic_var_to_df = {"sex":"sex_label",
                   "age":"age",
                   "view":"ViewPosition_label",
                   "race":"race_label",
                   "finding":"disease_label",
                   "pe_finding":"disease_label",
                   }


eps = 1e-8

class CheXMaskDataGenerator(Dataset):

    def __init__(self, 
                 inputpath: str = "/data2/rmehta3/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_files_preprocessed/", # path to folder containing input images
                 labelpath: str = "/data2/rmehta3/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_segmentation_preprocessed/", # path to folder containing input images
                 augment: bool = False, # data augmentation if true
                 scale_range: float = 0.0, # if not 0, then random scaling in range of 1 +- scale_range   
                 rotation_degree: float = 0.0, # if not 0, then random rotation in range of +- rotation_degree
                 csvpath: str = "/data2/rmehta3/datasets/chest_xray/test_pe_cm.csv", # full path to csv file
                 rca_threshold: float = 0.8,
                 subset: str = "test",
                 parents_x: List = ["sex", "race", "age", "finding", "view"],
                 concat_pa: bool = False,
                 add_dummy_dim: bool = True, # add dummy dimension of 1 to variables defined in dummy_var
                 dummy_var: List = ["finding", "view"], # add dummy dimension for softmax_centered trick in PGM 
                 ):

        self.concat_pa = concat_pa
        self.parents_x = parents_x
        self.add_dummy_dim = add_dummy_dim
        self.dummy_var = dummy_var
    
        self.data = pd.read_csv(csvpath)
        if rca_threshold>0.0:
            self.data = self.data[self.data['Dice RCA (Mean)']>=rca_threshold]
        self.data = self.data.reset_index(drop=True)

        self.inputpath = inputpath
        self.labelpath = labelpath

        self.augment = augment
        self.subset = subset
        self.scale_range = scale_range
        self.rotation_degree = rotation_degree

        # this is required because it is used in counterfactual training
        self.samples = {}
        for i in parents_x:
            sample = torch.tensor(self.data[mimic_var_to_df[i]])            
            if i == "age":
                self.samples[i] = sample.float() / 100 * 2 -1 
            else:
                n_classes = mimic_var_shape[i]
                if i in dummy_var and add_dummy_dim:
                    n_classes += 1
                self.samples[i] = F.one_hot(sample,num_classes=n_classes).squeeze()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        dicom_id = self.data.iloc[idx]["dicom_id"]

        # Generate data
        X = cv2.imread(os.path.join(self.inputpath,dicom_id)+'.jpg') # (256,256,3)
        # X = cv2.imread(os.path.join(self.inputpath,dicom_id)+'.png') # (256,256,3)
        y = cv2.imread(os.path.join(self.labelpath,dicom_id)+'.png') # (256,256,3) # left lung, right lung, heart

        # in appropriate numpy data type
        X = X.astype('float32')                
        y = y.astype('uint8')
        y[y>0] = 1 # making sure that its binary label per channel

        # convert into tensors
        X = torch.from_numpy(X)

        # X = X / X.max()
        X = (X*2/255) - 1 #[-1,1]

        y = torch.from_numpy(y)

        X = X.permute(2,0,1) # (3,256,256)
        X = X[0,:,:] # (256,256)

        y = y.permute(2,0,1) # (3,256,256)

        # # convert into tv_tensors
        X_t = tv_tensors.Image(X)
        y_t = tv_tensors.Mask(y)

        # perform dataaugmentation: dataaugmentations 
        # similar to https://github.com/baumgach/PHiSeg-code/blob/master/data/batch_provider.py#L140        
        if self.subset=="train" and self.augment:
            transforms = v2.Compose([
                # v2.RandomVerticalFlip(p=0.5),
                # v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=self.rotation_degree),
                v2.RandomResizedCrop(size=(X.shape[0],X.shape[1]),scale=(1-self.scale_range,1+self.scale_range))
                ])
            
            X_t, y_t = transforms(X_t, y_t)

        sample = {}
        sample["x"] = X_t
        sample["y"] = y_t
        sample["dicom_id"] = dicom_id

        for k in self.parents_x:
            sample[k] = torch.tensor([self.data.iloc[idx][mimic_var_to_df[k]]])
            if k == "age":
                sample[k] = sample[k] / 100 * 2 - 1  # [-1,1]
            else:
                num_classes = mimic_var_shape[k]
                if k in self.dummy_var and self.add_dummy_dim:
                    num_classes += 1 # extra dimension for gumble softmax and softmax_centered transform
                sample[k] = F.one_hot(sample[k], num_classes=num_classes).squeeze()

        if self.concat_pa:
            sample["pa"] = torch.cat([sample[k] for k in self.parents_x], dim=0,)

        return sample


def mimic(args: Hparams, augment=True) -> Dict[str, CheXMaskDataGenerator]:

    csv_dict = {"train":args.train_csv, "test":args.test_csv, "valid":args.valid_csv}
    datasets = {}
    for split in ["train", "valid", "test"]:
        datasets[split] = CheXMaskDataGenerator(
                 inputpath = args.inputpath,
                 labelpath = args.labelpath,
                 augment = True if (split=="train" and augment==True) else False,
                 scale_range = args.scale_range, 
                 rotation_degree = args.rotation_degree,
                 csvpath = os.path.join(args.csvpath,csv_dict[split]),
                 rca_threshold = args.rca_threshold,
                 subset = split,
                 parents_x = args.parents_x,
                 concat_pa = (False if not hasattr(args, "concat_pa") else args.concat_pa),
                 add_dummy_dim = args.add_dummy_dim, # add dummy dimension of 1 to variables defined in dummy_var
                 dummy_var = args.dummy_var, # add dummy dimension for softmax_centered trick in PGM 
        )    
    return datasets
