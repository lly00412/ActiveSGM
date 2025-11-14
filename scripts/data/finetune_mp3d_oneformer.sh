#!/bin/bash
##################################################
### This script is to run the full NARUTO system 
### (active planning and active ray sampling) 
###  on the Replica dataset.
##################################################

# Input arguments
EXP="generate_finetune_data" # config in configs/{DATASET}/{scene}/{EXP}.py will be loaded
GPU_ID=0

export CUDA_VISIBLE_DEVICES=${GPU_ID}
PROJ_DIR=${PWD}
DATASET=MP3D
# RESULT_DIR=${PROJ_DIR}/results/tmp

##################################################
### Scenes
###     choose one or all of the scenes
##################################################
scenes=(room0 room1 office0 office1 office2)

##################################################
### Main
###     Run for selected scenes for N trials
##################################################
CFG=configs/${DATASET}/${EXP}.py

python src/data/generate_finetune_data_Replica.py \
--seed 0 \
--result_dir ./data/tmp/generate_nvs_semantic/ \
--cfg ${CFG} \
--enable_vis 0

python src/data/finetune_oneformer_MP3D.py \
--seed 0 \
--result_dir ./data/checkpoint/oneformer/ \
--cfg ${CFG} \
--enable_vis 0
