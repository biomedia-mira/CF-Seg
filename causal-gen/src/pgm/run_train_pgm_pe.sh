#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_pgm.py --dataset mimic_pe \
                                           --seed 42 \
                                           --inputpath /vol/biomedic3/rmehta3/CheXMask/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_files_preprocessed/ \
                                           --labelpath /vol/biomedic3/rmehta3/CheXMask/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_segmentation_preprocessed/ \
                                           --csvpath /vol/biomedic3/rmehta3/CheXMask/datasets/chest_xray/ \
                                           --test_csv test_pe_only.csv \
                                           --valid_csv valid_pe_only.csv \
                                           --rca_threshold 0.0 \
                                           --epochs 150 \
                                           --bs 32 \
                                           --lr 0.0001 \
                                           --wd 0.1 \
                                           --input_res 256 \
                                           --input_channels 1 \
                                           --scale_range 0.1 \
                                           --rotation_degree 10.0 \
                                           --eval_freq 1 \
                                           --parents_x sex age view pe_finding race \
                                           --dummy_var view pe_finding \
                                           --exp_name pgm_mimic_train_pe_only_100 \
                                           --train_csv train_pe_only.csv \
                                           --add_dummy_dim \
                                           --deterministic