# ActiveSGM: Semantics-driven Active Mapping

This is the Python implementation of the ActiveSGM with SplaTAM backbone. (**Understanding while Exploring:
Semantics-driven Active Mapping**. Published at Neurips 2025) [[Paper](https://arxiv.org/abs/2506.00225)]

## Environment

### Installation

We provide scripts to create the conda environment, and recommend running ActiveSGM with Python 3.8 and CUDA 11.7 or CUDA 12.1. Please modify the scripts as needed to match your GPU and CUDA version.

```
# Download
git clone --recursive https://github.com/lly00412/ActiveSGM

# Build conda environment
cd ActiveSGM
bash scripts/installation/conda_env/build_sem.sh
```
### Build cuda tool for semantic rendering

#### dense-channel-rasterization
```
# clone from github
git clone -b liyan/dev --single-branch https://github.com/lly00412/semantic-gaussians.git third_parties/channel_rasterization

# go to the submodule directory
cd ./third_parties/channel_rasterization/channel-rasterization/cuda_rasterizer

# modify config.h base on number of class
NUM_CHANNELS {num of class} // Default 3

# install the cuda tool
cd ../..
python setup.py install
pip install .
```

#### sparse-channel-rasterization
```
# clone from github
git clone -b hairong/sparse_ver --single-branch https://github.com/lly00412/semantic-gaussians.git third_parties/sparse_channel_rasterization

# go to the submodule directory
cd ./third_parties/sparse_channel_rasterizationn/sparse-channel-rasterizationncuda_rasterizer

# modify config.h base on number of class and number of logits to keep
NUM_CHANNELS {num of class} // Default Replica: 102 MP3D:41
TOP_K_LOGITS_CHANNELS {number of logits to keep} // Default 16  

# install the cuda tool
cd ../..
python setup.py install
pip install .
```

## Data Preparation

### Dataset download
We run the experiments on [Replica](https://github.com/facebookresearch/Replica-Dataset/tree/main) and [Matterport3D](https://niessner.github.io/Matterport/)(MP3D) dataset using Habitat simulator, please follow the instruction of [ActiveGAMER](https://github.com/oppo-us-research/ActiveGAMER) to download these two datasets.

### Semantic mesh filtering for Matterport3D
We use Chamfer distance to remove floaters and generate clean semantic ground-truth meshes for MP3D scenes. Please run the following code before evaluation, and update the mesh file paths accordingly before running.```
```
python src/data/filter_mesh_mp3d.py
```

### Generate finetuning data for OneFormer

We provide fine-tuned OneFormer checkpoints for [Replica](https://huggingface.co/lly00412/oneformer-replica-finetune) and [MP3D](https://huggingface.co/lly00412/oneformer-mp3d-finetune). If you would like to run ActiveSGM on your own data, we also include configuration files and scripts for generating finetuning data.

We use [generate_finetune_data.py](https://github.com/lly00412/ActiveSGM/blob/main/configs/Replica/generate_finetune_data.py) as the configuration to generate semantic observation via Habitat simulator.
To finetune OneFormer, please run the following script:
```
# Modify the custom data folder before running
bash scripts/finetune_mp3d_oneformer.sh
```

## Training

We train ActiveSGM on two NVIDIA RTX A6000 GPUs. 
GPU 0 ("device") is used for keyframe mapping and path planning, 
while GPU 1 ("semantic_device") handles the OneFormer interface and semantic rendering. 
You can modify the "device" and "semantic_device" fields in the configuration files to assign these tasks to different GPUs as needed.

```
# Run ActiveSGM on Replica
bash scripts/activesgm/run_replica.sh {SCENE} {NUM_RUN} {EXP} {ENABLE_VIS} {GPU_ID}

# Run ActiveSGM on Replica office0
bash scripts/activesgm/run_replica.sh office0 1 ActiveSem 0 0,1

# Run Splatam
bash scripts/activesgm/run_replica.sh office0 1 predefine 0 0,1

# Run SGS-SLAM
bash scripts/activesgm/run_replica.sh office0 1 sgsslam 0 0,1
```

## Evaluation

We evaluate ActiveSGM for 3D reconstruction, Semantic Segmentation and Novel View Synthesis.
```
# Evaluate 3D reconstruction
bash scripts/evaluation/eval_replica_3d.sh office0 1 ActiveSem 0 0,1

# Evaluate semantic segmentation
bash scripts/evaluation/eval_replica_semantic.sh office0 1 ActiveSem 0 0 0 final

# Evaluate novel view synthesis
bash scripts/evaluation/eval_replica_nvs_result.sh office0 1 ActiveSem 0 0,1
```

## Citation

```
@inproceedings{chen2025understanding,
  title={Understanding while Exploring: Semantics-driven Active Mapping},
  author={Chen, Liyan and Zhan, Huangying and Yin, Hairong and Xu, Yi and Mordohai, Philippos},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```

## Acknowledgement
We sincerely thank the owners of the following open source projects, which are used by our released codes:
[HabitatSim](https://github.com/facebookresearch/habitat-sim), 
[ActiveGAMER](https://github.com/oppo-us-research/ActiveGAMER), 
[OneFormer](https://github.com/SHI-Labs/OneFormer),
[SplaTAM](https://github.com/spla-tam/SplaTAM),
[Semantic Gaussians](https://github.com/sharinka0715/semantic-gaussians),
[SGS-SLAM](https://github.com/ShuhongLL/SGS-SLAM).