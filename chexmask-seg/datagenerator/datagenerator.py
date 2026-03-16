import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.datasets as datasets
from torchvision import tv_tensors
from torchvision.transforms import v2
import numpy as np
import random
import scipy.ndimage as ndimage
import h5py
from os.path import join
import yaml
import glob
import SimpleITK as sitk
import nibabel
import pandas as pd
import cv2
import os

from datagenerator.utils import pad_array, percentile_clip, value_clip, normalize_intensity


###############################
### CheXMask DataGenerator
###############################

class CheXMaskDataGenerator(Dataset):

    def __init__(self, 
                 inputpath: str = "/data2/rmehta3/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_files_preprocessed/", # path to folder containing input images
                 labelpath: str = "/data2/rmehta3/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_segmentation_preprocessed/", # path to folder containing input images
                 augment: bool = False, # data augmentation if true
                 scale_range: float = 0.0, # if not 0, then random scaling in range of 1 +- scale_range   
                 rotation_degree: float = 0.0, # if not 0, then random rotation in range of +- rotation_degree
                 csvpath: str = "/data2/rmehta3/datasets/chest_xray/test_pe_only.csv", # full path to csv file
                 rca_threshold: float = 0.8,
                 subset: str = "test",
                 ):
        
        self.data = pd.read_csv(csvpath)
        self.data = self.data[self.data['Dice RCA (Mean)']>=rca_threshold]

        self.inputpath = inputpath
        self.labelpath = labelpath

        self.augment = augment
        self.subset = subset
        self.scale_range = scale_range
        self.rotation_degree = rotation_degree

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
           
        age = self.data.iloc[idx]["age"]
        sex = self.data.iloc[idx]["sex_label"] # male:0, female:1
        race = self.data.iloc[idx]["race_label"] # white:0, asian:1, black:2
        vp = self.data.iloc[idx]["ViewPosition_label"] # AP: 0, PA: 1
        disease = self.data.iloc[idx]["disease_label"] # NoFinding:0, PF:1, CM:2, PF&CM:3 
        
        dicom_id = self.data.iloc[idx]["dicom_id"]

        # Generate data
        # cv2.imread(os.path.join(inpbasedir,fold,f))
        X = cv2.imread(os.path.join(self.inputpath,dicom_id)+'.jpg') # (256,256,3)
        y = cv2.imread(os.path.join(self.labelpath,dicom_id)+'.png') # (256,256,3)


        # in appropriate numpy data type
        X = X.astype('float32')                
        y = y.astype('uint8')
        y[y>0] = 1

        # convert into tensors
        X = torch.from_numpy(X)
        X = X / X.max()
        y = torch.from_numpy(y)

        X = X.permute(2,0,1) # (3,256,256) 
        y = y.permute(2,0,1) # (3,256,256)

        # # convert into tv_tensors
        X_t = tv_tensors.Image(X)
        y_t = tv_tensors.Mask(y)

        # perform dataaugmentation: dataaugmentations similar to https://github.com/baumgach/PHiSeg-code/blob/master/data/batch_provider.py#L140        
        if self.subset=="train" and self.augment:
            transforms = v2.Compose([
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=self.rotation_degree),
                v2.RandomResizedCrop(size=(X.shape[1],X.shape[2]),scale=(1-self.scale_range,1+self.scale_range))
                ])
            
            X_t, y_t = transforms(X_t, y_t)

        return {'input':X_t, 'output':y_t, 'age':age, 'sex':sex, 'race':race, 'vp':vp, 'disease':disease, 'dicom_id':dicom_id}
