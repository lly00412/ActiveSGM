#!/bin/bash
##################################################
### This script is to run the full NARUTO system
### (active planning and active ray sampling)
###  on the Replica dataset.
##################################################

CUDA_LAUNCH_BLOCKING=1

# Input arguments
scene=${1:-office0}
num_run=${2:-1}
EXP=${3:-ActiveSem} # config in configs/{DATASET}/{scene}/{EXP}.py will be loaded
ENABLE_VIS=${4:-0}
GPU_ID=${5:-0}
STEP=${6:-0}
STAGE=${7:-final}

#export CUDA_VISIBLE_DEVICES=${GPU_ID}
export CUDA_VISIBLE_DEVICES=0,1
PROJ_DIR=${PWD}
#DATASET=Replica
DATASET=MP3D
RESULT_DIR=${PROJ_DIR}/results/

##################################################
### Random Seed
###     also used to initialize agent pose
###     from indexing the pose in Replica SLAM
###     trajectory.
##################################################
seeds=(0 500 1000 1500 1999)
seeds=("${seeds[@]:0:$num_run}")

##################################################
### Scenes
###     choose one or all of the scenes
##################################################
scenes=(room0 room1 room2 office0 office1 office2 office3 office4)
# Check if the input argument is 'all'
if [ "$scene" == "all" ]; then
    selected_scenes=${scenes[@]} # Copy all scenes
else
    selected_scenes=($scene) # Assign the matching scene
fi

##################################################
### Main
###     Run for selected scenes for N trials
##################################################
for scene in $selected_scenes
do
    for i in "${!seeds[@]}"; do
        seed=${seeds[$i]}

        ### create result folder ###
        result_dir=${RESULT_DIR}/${DATASET}/$scene/${EXP}/run_${i}
#        mkdir -p ${result_dir}

        ### run experiment ###
        CFG=configs/${DATASET}/${scene}/${EXP}.py
        python src/visualization/vis_semantic.py --cfg ${CFG} \
                                                --seed ${seed} \
                                                --result_dir ${result_dir} \
                                                --enable_vis ${ENABLE_VIS} \
                                                --stage $STAGE \
                                                --step ${STEP}


    done
done
