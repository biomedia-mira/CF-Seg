# CF-Seg
CF-Seg: Counterfactuals Meet Segmentation

## Causal-Gen
causal-gen: causal generative model based on HVAE. Modified from https://github.com/biomedia-mira/causal-gen

- download all checkpoints using the following link: https://drive.google.com/file/d/13lCVYRnQco7G10t3Rwln6R1fLoRGIvQf/view?usp=sharing
- put above checkpoints at: causal-gen/checkpoints/

### Training
- Train PGM: causal-gen/src/pgm/run_train_pgm_pe.sh
- Train HVAE: causal-gen/src/run_train_hvae.sh
- Train Soft Counterfactual Fine Tuning: causal-gen/src/pgm/run_train_cf_soft_pe.sh

### Inference
- Use provided checkpoints to save counterfactual images: causal-gen/src/pgm/run_save_cf_pe.sh

## CheXMask-Seg
chexmask-seg: simple U-Net trained on CheXMask dataset for lung segmentation. 

- download trained model weights using the following link: https://drive.google.com/file/d/1pHkWmLfz8cOLLjO0-tNoFY6egKeff21g/view?usp=sharing
- put above checkpoints at: chexmask-seg/checkpoints/

### Training
- train model: chexmask-seg/run_1.sh

## Dataset Download

All dataset preparation code is provided in: datasets_preparation/
- run CheXMask-MIMIC-merge-dataframes.ipynb to generate csv files. Note that it expects csv files from the orignal MIMIC datasets.
- run CheXMask-MIMIC-DataPreparation.py to resize all the MIMIC images to 1024x1024 resolution. Note that it expects original MIMIC images.
- run CheXMask-MIMIC-Seg-Generation.py to generate CheXMask masks for MIMIC in 1024x1024 resolution. Note that it expects original CheXMask csv files.
- run CheXMask-MIMIC-Resize-256x256.py to resize all the MIMIC images and CheXMask masks to 256x256 resolution. Note that it expects that images are already stored in 1024x1024 resolution using previous scripts.

Required CSV files to run Causal-Gen and CheXMask-Seg are provided in: datasets/mimic_csv/

Download pre-processed (and resized to 256x256) MIMIC dataset using the following link: https://drive.google.com/file/d/1_yf6hwXlmPOj74kT7_mUe91L74B5KrC3/view?usp=sharing

Download pre-processed (and resized to 256x256) CheXMask for MIMIC dataset using the following link: https://drive.google.com/file/d/1TeTbTXojBgZO1syXRDAFvDszfkwEba1r/view?usp=sharing

Download our expert defined Lung Segmentation for a selected of MIMIC dataset (256x256) using the following link: https://drive.google.com/file/d/1ohJzMnlB_NJ1WGWfx5us3DdNxZUep71g/view?usp=sharing

# Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{mehta2025cf,
  title={Cf-seg: Counterfactuals meet segmentation},
  author={Mehta, Raghav and De Sousa Ribeiro, Fabio and Xia, Tian and Roschewitz, Melanie and Santhirasekaram, Ainkaran and Marshall, Dominic C and Glocker, Ben},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={117--127},
  year={2025},
  organization={Springer}
}