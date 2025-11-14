import os
import matplotlib.cm as cm
from plyfile import PlyData, PlyElement
import torch
import numpy as np
from third_parties.splatam.utils.common_utils import params2cpu
from src.utils.general_utils import create_class_colormap,apply_colormap

def save_params_ckpt(output_params, output_variables, output_dir, time_idx):
    # Convert to CPU Numpy Arrays
    to_save = params2cpu(output_params)

    # also save semantic info
    for k in ['seman_cls_ids']:
        if isinstance(output_variables[k], torch.Tensor):
            to_save[k] = output_variables[k].detach().cpu().contiguous().numpy()
        else:
            to_save[k] = output_variables[k]

    # Save the Parameters containing the Gaussian Trajectories
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params"+str(time_idx)+".npz")
    np.savez(save_path, **to_save)

def save_rgb_ply(params,ckpt_output_dir, time_idx):
    os.makedirs(ckpt_output_dir, exist_ok=True)
    ply_name = f"rgb_GS_{time_idx:04}.ply"
    ply_savepath = os.path.join(ckpt_output_dir, ply_name)

    rgbs = params['rgb_colors'].detach().cpu().contiguous().numpy()
    opacities = params['logit_opacities'].detach().cpu().contiguous().numpy()
    write_ply_file(rgbs, opacities, params, ply_savepath)

def save_semantic_ply(params,variables, ckpt_output_dir, time_idx, n_cls=150, colormap=None):
    os.makedirs(ckpt_output_dir, exist_ok=True)
    ply_name = f"semantic_GS_{time_idx:04}.ply"
    ply_savepath = os.path.join(ckpt_output_dir, ply_name)

    class_ids_indices = params['semantic_logits'].argmax(-1).cpu().numpy()
    topk_class = variables['seman_cls_ids'].cpu().numpy()
    class_ids = topk_class[np.arange(topk_class.shape[0]),class_ids_indices]
    if colormap == None:
        sem_colormap = create_class_colormap(n_cls)
    else:
        sem_colormap =  colormap
    sem_rgbs = apply_colormap(class_ids, sem_colormap) / 255.
    # sem_opacities = self.params['logit_opacities_seman'].detach().cpu().contiguous().numpy()
    sem_opacities = params['logit_opacities'].detach().cpu().contiguous().numpy()
    write_ply_file(sem_rgbs, sem_opacities, params, ply_savepath)

def write_ply_file(rgbs, opacities, params, ply_savepath):
    means = params['means3D'].detach().cpu().contiguous().numpy()
    rotations = params['unnorm_rotations'].detach().cpu().contiguous().numpy()
    scales = params['log_scales'].detach().cpu().contiguous().numpy()
    normals = np.zeros_like(means)
    C0 = 0.28209479177387814
    colors = (rgbs - 0.5) / C0
    if scales.shape[1] == 1:
        scales = np.tile(scales, (1, 3))
    attrs = ['x', 'y', 'z',
             'nx', 'ny', 'nz',
             'f_dc_0', 'f_dc_1', 'f_dc_2',
             'opacity',
             'scale_0', 'scale_1', 'scale_2',
             'rot_0', 'rot_1', 'rot_2', 'rot_3', ]
    dtype_full = [(attribute, 'f4') for attribute in attrs]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = np.concatenate((means, normals, colors, opacities, scales, rotations), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(ply_savepath)
