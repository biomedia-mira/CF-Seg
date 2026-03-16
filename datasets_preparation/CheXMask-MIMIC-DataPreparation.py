import numpy as np
import os
import cv2 
import pandas as pd
import gc
from tqdm import tqdm

def remove_padding(img):
    
    gray = 255*(img > 1) # To invert the text to white
    coords = cv2.findNonZero(gray) # Find all non-zero points (text)

    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
    cropimg = img[y:y+h, x:x+w] # Crop the image - note we do this on the original image
    
    pad_left, pad_top, pad_right, pad_bottom = x, y, img.shape[1] - x - w, img.shape[0] - y - h
    
    return cropimg, pad_left, pad_top, pad_right, pad_bottom


def pad_to_square_centered(img):
    h, w = img.shape[:2]
    longest_edge = max(h, w)
    result = np.zeros((longest_edge, longest_edge), dtype=img.dtype)
    pad_h = (longest_edge - h) // 2
    pad_w = (longest_edge - w) // 2
    result[pad_h:pad_h + h, pad_w:pad_w + w] = img

    pad_left = pad_w
    pad_top = pad_h
    pad_right = longest_edge - w - pad_w
    pad_bottom = longest_edge - h - pad_h

    return result, pad_left, pad_top, pad_right, pad_bottom

#########

df = pd.read_csv("/vol/biodata/data/chest_xray/mimic-cxr-jpg/mimic-cxr-2.0.0-metadata.csv.gz")
base_path = "/vol/biodata/data/chest_xray/mimic-cxr-jpg/files/"
output_path = "/data2/rmehta3/datasets/chest_xray/mimic-chexmask-jpg-1024x1024-new/CheXMask_files_preprocessed/"


try:
    os.makedirs(output_path,exist_ok=True)
except:
    pass


#############

progress_bar = tqdm(df.iterrows(), total=len(df), ncols=70)

with open(output_path+'mimic_padd_added.csv', '+w') as f:

    f.write("filename,height,width,pad_left_rem,pad_top_rem,pad_right_rem,pad_bottom_rem,pad_left,pad_top,pad_right,pad_bottom \n")
    
    with open(output_path+"mimic_errors.csv", "+w") as errorf:

        for index, row in progress_bar:
                        
            subject_id = 'p' + str(row['subject_id'])
            study_id = 's' + str(row['study_id'])
            dicom_id = row['dicom_id'] + '.jpg'
            dicom_id_png = row['dicom_id'] + '.png'
            
            sub = subject_id[:3]

            try:
                os.mkdir(os.path.join(output_path, sub))
            except:
                pass 
            
            subject = os.path.join(output_path, sub, subject_id)
            
            try:
                os.mkdir(subject)
            except:
                pass 
            
            study = os.path.join(subject, study_id)
            
            try:
                os.mkdir(study)
            except:
                pass
            
            img_path = os.path.join(base_path, sub, subject_id, study_id, dicom_id)
            out_path = os.path.join(output_path, dicom_id_png)
            
            try:
                img = cv2.imread(img_path, 0)
                
                h, w = img.shape[:2]
                    
                # remove padding
                img2, pad_left, pad_top, pad_right, pad_bottom = remove_padding(img)
                reshaped, pad_left2, pad_top2, pad_right2, pad_bottom2 = pad_to_square_centered(img2)
                # scale to 1024x1024
                scaled = cv2.resize(reshaped, (1024, 1024))
                cv2.imwrite(out_path, scaled)

                del img
                del img2
                del scaled           
                
                gc.collect()
                
                image = os.path.join(sub, subject_id, study_id, dicom_id)
                
                outline = "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d \n"%(image, h, w, pad_left, pad_top, pad_right, pad_bottom, pad_left2, pad_top2, pad_right2, pad_bottom2)
                f.write(outline)
                
            except:
                errorf.write(image + " \n")
                print("Error with image %s"%image)       
