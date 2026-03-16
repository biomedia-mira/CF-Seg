#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_cf.py   --exp_name soft_cf_1_0_mimic_train_100_lr_0_001_wd_0_001_lr_lag_0_001_alpha_0_1_beta_3_pe_only \
                                            --dataset mimic_pe \
                                            --seed 42 \
                                            --deterministic \
                                            --inputpath /vol/biomedic3/rmehta3/CheXMask/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_files_preprocessed/ \
                                            --labelpath /vol/biomedic3/rmehta3/CheXMask/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_segmentation_preprocessed/ \
                                            --csvpath /vol/biomedic3/rmehta3/CheXMask/datasets/chest_xray/ \
                                            --test_csv test_pe_only.csv \
                                            --valid_csv valid_pe_only.csv \
                                            --pgm_predictor_path /vol/biomedic3/rmehta3/CheXMask/code/causal-gen-23-June/checkpoints/s_a_v_p_r/pgm_mimic_train_pe_only_100/checkpoint.pt \
                                            --vae_path /vol/biomedic3/rmehta3/CheXMask/code/causal-gen-23-June/checkpoints/s_a_v_p_r/hvae_mimic_train_100_beta_3_lr_0_001_var_0_01_wd_0_001_pe_only/checkpoint.pt \
                                            --train_csv train_pe_only.csv \
                                            --bs 7 \
                                            --lr 0.001 \
                                            --wd 0.001 \
                                            --alpha 0.1 \
                                            --lr_lagrange 0.001 \
                                            --epochs 4 \
                                            --eval_freq 1 \
                                            --soft_cf \
                                            --do_multip 1.0
