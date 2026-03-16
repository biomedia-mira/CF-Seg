import cv2
import numpy as np
import os
from tqdm import tqdm

inpbasedir = "/data2/rmehta3/datasets/chest_xray/mimic-chexmask-jpg-1024x1024/"
outpbasedir = "/data2/rmehta3/datasets/chest_xray/mimic-chexmask-jpg-256x256/"

folders = ["CheXMask_segmentation_preprocessed", "CheXMask_files_preprocessed"]

interp = [cv2.INTER_NEAREST, cv2.INTER_LINEAR]

f_num = 0

for fold in folders:

    files = os.listdir(os.path.join(inpbasedir,fold))
    os.makedirs(os.path.join(outpbasedir,fold), exist_ok=True)

    for i, f in enumerate(tqdm(files)):

        inp_file = cv2.imread(os.path.join(inpbasedir,fold,f))
        scaled = cv2.resize(inp_file, (256, 256), interpolation=interp[f_num])
        cv2.imwrite(os.path.join(outpbasedir,fold,f), scaled)

    f_num += 1
