#!/bin/bash

python save_cf.py \
    --dataset mimic_pe \
    --seed 42 \
    --deterministic \
    --load_path /vol/biomedic3/rmehta3/CheXMask/code/causal-gen-23-June/checkpoints/s_a_v_p_r/soft_cf_1_0_mimic_train_100_lr_0_001_wd_0_001_lr_lag_0_001_alpha_0_1_beta_3_pe_only/881_checkpoint.pt \
    --save_path /vol/biomedic3/rmehta3/CheXMask/code/causal-gen-23-June/checkpoints/ \
    --save_folder counterfactual_images_train_100_pe_only \
    --inputpath /vol/biomedic3/rmehta3/CheXMask/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_files_preprocessed/ \
    --labelpath /vol/biomedic3/rmehta3/CheXMask/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_segmentation_preprocessed/ \
    --csvpath /vol/biomedic3/rmehta3/CheXMask/datasets/chest_xray/ \
    --train_csv train_pe_only.csv \
    --valid_csv valid_pe_only.csv \
    --test_csv test_pe_only.csv \
    --save_set test \
    --rca_threshold 0.0 \
    --cf_particles 1 \
    --interven_variables pe_finding \
    --pgm_predictor_path /vol/biomedic3/rmehta3/CheXMask/code/causal-gen-23-June/checkpoints/s_a_v_p_r/pgm_mimic_train_pe_only_100/checkpoint.pt \
    --vae_path /vol/biomedic3/rmehta3/CheXMask/code/causal-gen-23-June/checkpoints/s_a_v_p_r/hvae_mimic_train_100_beta_3_lr_0_001_var_0_01_wd_0_001_pe_only/checkpoint.pt 