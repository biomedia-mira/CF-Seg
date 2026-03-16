#!/bin/bash

python main.py --seed 42 \
              --hps mimic_pe \
              --inputpath /vol/biomedic3/rmehta3/CheXMask/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_files_preprocessed/ \
              --labelpath /vol/biomedic3/rmehta3/CheXMask/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_segmentation_preprocessed/ \
              --csvpath /vol/biomedic3/rmehta3/CheXMask/datasets/chest_xray/ \
              --dataset mimic_pe \
              --parents_x sex age view pe_finding race \
              --dummy_var view pe_finding \
              --context_dim 12 \
              --add_dummy_dim \
              --deterministic \
              --concat_pa \
              --cond_prior \
              --epochs 200 \
              --rca_threshold 0.0 \
              --eval_freq 1 \
              --beta 3.0 \
              --lr 0.001 \
              --std_init 0.01 \
              --wd 0.001 \
              --test_csv test_pe_only.csv \
              --valid_csv valid_pe_only.csv \
              --exp_name hvae_mimic_train_100_beta_3_lr_0_001_var_0_01_wd_0_001_pe_only \
              --train_csv train_pe_only.csv