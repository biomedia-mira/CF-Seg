import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F

from os.path import join

plt.rcParams["savefig.bbox"] = 'tight'

def show_samples(images: np.array, # (B,C,H,W)
                 row_headers: list =[], # (B)
                 column_headers: list = [], # (C)
                 epoch: int = 0, 
                 save_dir: str = ""):

    assert images.shape[0] == len(row_headers)
    assert images.shape[1] == len(column_headers)

    subplot_width = 2
    subplot_height = 2

    # Create a figure and axis objects
    fig, axs = plt.subplots(len(row_headers), len(column_headers), figsize=(subplot_width*len(column_headers), subplot_height*len(row_headers)))  

    # Iterate through each image and display it
    for i in range(len(row_headers)):
        for j in range(len(column_headers)):            

            axs[i, j].imshow(images[i,j,...], cmap='gray')  # Assuming grayscale images, change cmap if necessary
            axs[i, j].axis('off')  # Turn off axis

            # Add header on top of each column
            if i == 0:
                axs[i, j].set_title(column_headers[j])
            # Add label on the side of each row
            if j == 0:
                axs[i, j].set_ylabel(row_headers[i], rotation=90, fontsize=12)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.savefig(join(save_dir ,"output_{:03d}.png".format(epoch)), dpi=300)   
