import numpy as np 
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt 
from tqdm import tqdm

###############

def get_RLE_from_mask(mask):
    mask = (mask / 255).astype(int)
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def get_mask_from_RLE(rle, height, width):
    runs = np.array([int(x) for x in rle.split()])
    starts = runs[::2]
    lengths = runs[1::2]

    mask = np.zeros((height * width), dtype=np.uint8)

    for start, length in zip(starts, lengths):
        start -= 1  
        end = start + length
        mask[start:end] = 255

    mask = mask.reshape((height, width))
    
    return mask


def getDenseMaskFromLandmarks(landmarks, imagesize = 1024):
    img = np.zeros([imagesize,imagesize])
    landmarks = landmarks.reshape(-1, 1, 2).astype('int')
    img = cv2.drawContours(img, [landmarks], -1, 255, -1)
    return img

##############

path = "/data2/rmehta3/datasets/chexmask-cxr-segmentation-data/0.4/Preprocessed/MIMIC-CXR-JPG.csv"
df = pd.read_csv(path)
columns = df.columns

output_path = "/data2/rmehta3/datasets/chest_xray/mimic-chexmask-jpg-1024x1024/CheXMask_segmentation_preprocessed/"
os.makedirs(output_path,exist_ok=True)

progress_bar = tqdm(df.iterrows(), total=len(df), ncols=70)

for index, row in progress_bar:

    landmarks = row["Landmarks"]
    height, width = row["Height"], row["Width"]

    # As landmarks are stored as a string, we need to convert it to a numpy array
    landmarks = eval(landmarks)
    landmarks = np.array(landmarks).reshape(-1, 2)

    rightLungLandmarks = landmarks[:44, :]
    leftLungLandmarks = landmarks[44:94, :]
    heartLandmarks = landmarks[94:, :]

    # get Mask RLE
    rightLungMask_RLE = row["Right Lung"]
    leftLungMask_RLE = row["Left Lung"]
    heartMask_RLE = row["Heart"]

    rightLungMask = get_mask_from_RLE(rightLungMask_RLE, height, width)
    leftLungMask = get_mask_from_RLE(leftLungMask_RLE, height, width)
    heartMask = get_mask_from_RLE(heartMask_RLE, height, width)


    # generate a combined mask
    mask = np.zeros([height, width, 3], dtype = np.uint8)
    mask[:, :, 0] = rightLungMask
    mask[:, :, 1] = leftLungMask
    mask[:, :, 2] = heartMask

    # save image
    out_path = os.path.join(output_path,row['dicom_id']+'.png')
    cv2.imwrite(out_path,mask)
