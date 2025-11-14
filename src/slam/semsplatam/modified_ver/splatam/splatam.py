import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torch_sparse.tensor import SparseTensor

from third_parties.splatam.utils.slam_helpers import (
transformed_params2rendervar,
 transformed_params2depthplussilhouette,
    transform_to_frame,l1_loss_v1,
)
from third_parties.splatam.utils.slam_external import calc_ssim
import matplotlib.pyplot as plt

from third_parties.splatam.utils.slam_external import build_rotation
from third_parties.splatam.utils.gs_external import update_params_and_optimizer, inverse_sigmoid,cat_params_to_optimizer, accumulate_mean2d_gradient
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from channel_rasterization import GaussianRasterizationSettings as Camera
from sparse_channel_rasterization import GaussianRasterizer as SEMRenderer_sparse
from sparse_channel_rasterization import GaussianRasterizationSettings as Camera_sparse

from src.slam.semsplatam.modified_ver.semantic.oneformer import positive_normalize

def setup_camera(w, h, k, w2c, near=0.01, far=100, num_channels=102):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
        debug=False,
        num_channels=num_channels,
    )
    return cam

def create_differentiable_sparse_tensor(dense_tensor, topk_indices, dense_shape):
    N = dense_tensor.shape[0]
    top_k = topk_indices.shape[1]

    row_indices = torch.arange(N).repeat_interleave(top_k).to(topk_indices.device)
    col_indices = topk_indices.reshape(-1)

    i_indices = torch.arange(N).repeat_interleave(top_k)
    j_indices = topk_indices.reshape(-1)

    topk_values = dense_tensor[i_indices, j_indices]
    sparse_coordinates = torch.stack([row_indices, col_indices], dim=0)

    sparse_tensor = torch.sparse_coo_tensor(
        sparse_coordinates,
        topk_values,
        size=dense_shape,
    )

    sparse_tensor = SparseTensor.from_torch_sparse_coo_tensor(sparse_tensor)

    return sparse_tensor
def get_pointcloud_with_seman(color, depth, seman, intrinsics, w2c, transform_pts=True,
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(),
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX) / FX
    yy = (y_grid - CY) / FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY) / 2)
            mean3_sq_dist = scale_gaussian ** 2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")

    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)  # (C, H, W) -> (H, W, C) -> (H * W, C)
    h,w = seman.shape[1], seman.shape[2]
    seman = torch.permute(seman, (1, 2, 0)).reshape(h*w, -1)  # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols, seman), -1)

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld

def initialize_optimizer(params, lrs_dict, tracking):
    # TODO: fix ValueError: can't optimize a non-leaf Tensor
    param_groups = []
    #param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    for k, v in params.items():
        param_groups.append({'params': [v], 'name': k, 'lr': lrs_dict[k]})
    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

def initialize_params_with_seman(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution, TOPK = 16):
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")

    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'semantic_logits': init_pt_cld[:, 6:],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    _, topk_indices = torch.topk(params['semantic_logits'], k=TOPK, dim=-1)
    dense_shape = params['semantic_logits'].shape
    seman_sparse = create_differentiable_sparse_tensor(params['semantic_logits'], topk_indices, dense_shape)

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v)

        if k in ['semantic_logits']:
            sparse_values = seman_sparse.coo()[2].reshape(dense_shape[0],TOPK)
            params[k] = torch.nn.Parameter(sparse_values.cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'seman_cls_ids': seman_sparse.coo()[1].reshape(dense_shape[0],TOPK),} # col index
    return params, variables

def initialize_first_timestep(dataset, seman, num_frames, scene_radius_depth_ratio,
                              mean_sq_dist_method, densify_dataset=None, gaussian_distribution=None, TOPK=16, num_classes=102):
    # Get RGB-D Data & Camera Parameters
    color, depth, intrinsics, pose = dataset[0]
    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255  # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    seman = seman.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    n_cls = seman.shape[0]

    H,W = color.shape[1], color.shape[2]
    if (seman.shape[1] != H) or (seman.shape[2] != W):
        seman = F.interpolate(seman.unsqueeze(0), (H, W), mode='bilinear')[0]

    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy(), num_channels=num_classes)

    if densify_dataset is not None:
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255  # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        H, W = color.shape[1], color.shape[2]
        if (seman.shape[1] != H) or (seman.shape[2] != W):
            seman = F.interpolate(seman.unsqueeze(0), (H, W), mode='bilinear')[0]
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(),
                                   w2c.detach().cpu().numpy(),num_channels=seman.shape[0])
    else:
        densify_intrinsics = intrinsics

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0)  # Mask out invalid depth values
    mask = mask.reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud_with_seman(color, depth, seman, densify_intrinsics, w2c,
                                                mask=mask, compute_mean_sq_dist=True,
                                                mean_sq_dist_method=mean_sq_dist_method)

    # Initialize Parameters
    params, variables = initialize_params_with_seman(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution, TOPK)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth) / scene_radius_depth_ratio
    variables['n_cls'] = n_cls

    if densify_dataset is not None:
        return params, variables, intrinsics, w2c, cam, densify_intrinsics, densify_cam
    else:
        return params, variables, intrinsics, w2c, cam

def initialize_new_params_with_seman(new_pt_cld, mean3_sq_dist, gaussian_distribution, TOPK=16):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'semantic_logits': new_pt_cld[:, 6:],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }
    _, topk_indices = torch.topk(params['semantic_logits'], k=TOPK, dim=-1)
    dense_shape = params['semantic_logits'].shape
    seman_sparse = create_differentiable_sparse_tensor(params['semantic_logits'], topk_indices, dense_shape)

    for k, v in params.items():
        # Check if value is already a torch tensor
        if k in ['semantic_logits']:
            sparse_values = seman_sparse.coo()[2].reshape(dense_shape[0], TOPK)
            params[k] = torch.nn.Parameter(sparse_values.float().contiguous().requires_grad_(True))
        else:
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'seman_cls_ids': seman_sparse.coo()[1].reshape(dense_shape[0],TOPK), }

    return params, variables


def add_new_gaussians_with_seman(params, variables, curr_data, sil_thres,
                      time_idx, mean_sq_dist_method, gaussian_distribution, TOPK=16):
    # Silhouette Rendering
    transformed_gaussians = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_gaussians)
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())

    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        new_pt_cld, mean3_sq_dist = get_pointcloud_with_seman(curr_data['im'], curr_data['depth'], curr_data['seman'],curr_data['intrinsics'],
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)
        new_params,new_variables = initialize_new_params_with_seman(new_pt_cld,mean3_sq_dist, gaussian_distribution,TOPK)
        #num_pts_prev = params['means3D'].shape[0]
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))

        variables['seman_cls_ids'] = torch.cat((variables['seman_cls_ids'], new_variables['seman_cls_ids']),dim=0)
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)

    return params, variables


def transformed_params2semrendervar(params, variables, transformed_gaussians, seen):
    # Check if Gaussians are Isotropic
    N,C = params['log_scales'].shape[0], variables['n_cls']
    topk = variables['seman_cls_ids'].shape[-1]
    row_index = torch.arange(N).repeat_interleave(topk).to(variables['seman_cls_ids'].device)
    seman_sparse = SparseTensor(row=row_index.view(-1),
                                col=variables['seman_cls_ids'].view(-1),
                                value=params['semantic_logits'].view(-1),
                                sparse_sizes=(N, C))

    if params['log_scales'].shape[1] == 1:
        log_scales = torch.tile(params['log_scales'][seen], (1, 3))
    else:
        log_scales = params['log_scales'][seen]

    # Initialize Render Variables
    rendervar = {
        'means3D': transformed_gaussians['means3D'][seen],
        'colors_precomp': seman_sparse.to_dense()[seen],
        'rotations': F.normalize(transformed_gaussians['unnorm_rotations'][seen]),
        'opacities': torch.sigmoid(params['logit_opacities'][seen]),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'][seen], requires_grad=True, device="cuda") + 0,
    }
    return rendervar

def transformed_params2semrendervar_sparse(params, transformed_gaussians, seen):
    if params['log_scales'].shape[1] == 1:
        log_scales = torch.tile(params['log_scales'][seen], (1, 3))
    else:
        log_scales = params['log_scales'][seen]

    # Initialize Render Variables
    rendervar = {
        'means3D': transformed_gaussians['means3D'][seen].detach(),
        'colors_precomp': params['semantic_logits'][seen],
        'rotations': F.normalize(transformed_gaussians['unnorm_rotations'][seen]).detach(),
        'opacities': torch.sigmoid(params['logit_opacities'][seen]).detach(),
        'scales': torch.exp(log_scales).detach(),
        'means2D': torch.zeros_like(params['means3D'][seen], requires_grad=True, device="cuda") + 0,
    }

    return rendervar

def set_camera_sparse(cam, cls_ids=None):
    cam = Camera_sparse(
        image_height=cam.image_height,
        image_width=cam.image_width,
        tanfovx=cam.tanfovx,
        tanfovy=cam.tanfovy,
        bg=cam.bg,
        scale_modifier=1.0,
        viewmatrix=cam.viewmatrix,
        projmatrix=cam.projmatrix,
        sh_degree=cam.sh_degree,
        campos=cam.campos,
        prefiltered=False,
        debug=False,
        num_channels=cam.num_channels,
        cls_ids=cls_ids.to(torch.int32)
    )
    return cam

def calc_cosine(tensor1, tensor2, dim=0,return_mean=True, required_normalize=True):
    if required_normalize:
        eps = 1e-8
        norm1 = torch.norm(tensor1, p=2, dim=0, keepdim=True) + eps
        norm2 = torch.norm(tensor2, p=2, dim=0, keepdim=True) + eps
    else:
        norm1 = tensor1
        norm2 = tensor2
    cosine = torch.nn.CosineSimilarity(dim=dim)
    if return_mean:
        return cosine(tensor1/norm1,tensor2/norm2).mean()
    else:
        return cosine(tensor1/norm1,tensor2/norm2)

def calc_kl(pred_dist,target_dist,reduction='batchmean',eps=1e-8):
    if len(pred_dist.shape) == 2:
        c,h = pred_dist.shape
        w = 1
    else:
        c,h,w = pred_dist.shape
    pred_dist = pred_dist.reshape(-1,h*w).permute(1,0)
    target_dist = target_dist.reshape(-1,h*w).permute(1,0)
    kl_loss = torch.nn.KLDivLoss(reduction=reduction)
    pred_dist = F.log_softmax(pred_dist+eps,dim=-1)
    return kl_loss(pred_dist,target_dist)

def calc_hellinger_distance(pred_dist,target_dist,eps=1e-8):
    if len(pred_dist.shape) == 2:
        c,h = pred_dist.shape
        w = 1
    else:
        c,h,w = pred_dist.shape

    pred_dist = pred_dist.reshape(-1,h*w).permute(1,0)
    target_dist = target_dist.reshape(-1,h*w).permute(1,0)
    pred_dist = torch.clamp(pred_dist,min=0.)
    target_dist = torch.clamp(target_dist, min=0)
    sqrt_pred = torch.sqrt(pred_dist)
    sqrt_target = torch.sqrt(target_dist)
    dist = torch.sqrt(0.5 * torch.sum((sqrt_pred- sqrt_target) ** 2, dim=-1))
    return dist.mean()

def calc_shannon_entropy(prob_dist,dim=-1):
    prob_dist = prob_dist.to(torch.float32)
    prob_dist = torch.clamp(prob_dist, min=0.001)
    prob_dist = positive_normalize(prob_dist,dim=dim, min=0)
    entropy = -torch.sum(prob_dist * torch.log(prob_dist),dim=dim)
    # entropy = torch.distributions.Categorical(prob_dist).entropy()
    return entropy

def calc_cross_entropy(pred_dist,target_dist,dim=0): # (c,h,w) (c,h,w)
    target_dist = target_dist
    pred_dist = pred_dist
    log_prob = F.log_softmax(pred_dist+1e-8, dim=dim)
    loss = -(target_dist * log_prob).sum(dim=dim).mean()
    return loss

def get_crop_mask(image: torch.Tensor, crop_size: int = 50):
    _, h, w = image.shape
    assert crop_size * 2 < h and crop_size * 2 < w, "Crop size too large!"
    mask = torch.zeros((h, w), dtype=torch.bool)
    mask[crop_size:h - crop_size, crop_size:w - crop_size] = True
    return mask

def get_uncert_mask(distribution: torch.Tensor, filter_pct=0.1, thres=3.0):
    entropy = calc_shannon_entropy(distribution.clone(),dim=0)
    thres = torch.quantile(entropy.view(-1), 1- filter_pct)
    mask = (entropy > thres) & (entropy > thres)  # (3.0 is entropy of uniform topk 16)
    return ~mask


def get_loss_with_seman(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1, ignore_outlier_depth_loss, tracking=False,
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None,
             lambda_hel = 0.8, lambda_cosine = 0.2):
    # Initialize Loss Dictionary
    losses = {}

    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                   gaussians_grad=False,
                                                   camera_grad=True)
    elif mapping:
        if do_ba:
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                       gaussians_grad=True,
                                                       camera_grad=True)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                       gaussians_grad=True,
                                                       camera_grad=False)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                   gaussians_grad=True,
                                                   camera_grad=False)

    # Initialize Render Variables
    rgb_rendervar = transformed_params2rendervar(params, transformed_gaussians)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_gaussians)
    # RGB Rendering
    rgb_rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rgb_rendervar)
    seen = radius>0
    variables['means2D'] = rgb_rendervar['means2D']  # Gradient only accum from colour render for densification

    # Depth & Silhouette Rendering
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = depth_sq - depth ** 2
    uncertainty = uncertainty.detach()

    # Semantic Rendering
    seman_rendervar = transformed_params2semrendervar_sparse(params, transformed_gaussians, seen)
    sparse_cam = set_camera_sparse(cam=curr_data['cam'],cls_ids=variables['seman_cls_ids'])
    seman_im, _, = SEMRenderer_sparse(raster_settings=sparse_cam)(**seman_rendervar)

    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10 * depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
    # Mask with presence silhouette mask (accounts for empty space)
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()

    # RGB Loss
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
    else:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))

    # Semantic Loss
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        n_channels = seman_im.shape[0]
        seman_mask = torch.tile(mask, (n_channels, 1, 1))
        seman_mask = seman_mask.detach()
        losses['seman'] = 1- calc_cosine(curr_data['seman'],seman_im,dim=0)[seman_mask].sum()
    elif tracking:
        losses['seman'] = 1- calc_cosine(curr_data['seman'],seman_im,dim=0).sum()
    else:
        if 'crop_mask' in curr_data.keys():

            ### when optimizing the global keyframe, crop the boundary pixels for uncertainty semantic info
             seman_im = seman_im[:,curr_data['crop_mask']]
             seman_pseudo = curr_data['seman'][:, curr_data['crop_mask']]
             losses['seman'] = lambda_hel * calc_hellinger_distance(pred_dist=seman_im, target_dist=seman_pseudo) + lambda_cosine * (1 - calc_cosine(seman_pseudo, seman_im, dim=0))
        else:
            losses['seman'] = lambda_hel * calc_hellinger_distance(pred_dist=seman_im,target_dist=curr_data['seman']) + lambda_cosine *(1-calc_cosine(curr_data['seman'],seman_im,dim=0))
            #losses['seman'] = lambda_hel * calc_kl(pred_dist=seman_im,target_dist=curr_data['seman']) + lambda_cosine * (1 - calc_cosine(curr_data['seman'], seman_im, dim=0))


    # Visualize the Diff Images
    if tracking and visualize_tracking_loss:
        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        weighted_render_im = im * color_mask
        weighted_im = curr_data['im'] * color_mask
        weighted_render_depth = depth * mask
        weighted_depth = curr_data['depth'] * mask
        diff_rgb = torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
        diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()
        viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[0, 0].imshow(viz_img)
        ax[0, 0].set_title("Weighted GT RGB")
        viz_render_img = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[1, 0].imshow(viz_render_img)
        ax[1, 0].set_title("Weighted Rendered RGB")
        ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[0, 1].set_title("Weighted GT Depth")
        ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[1, 1].set_title("Weighted Rendered Depth")
        ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
        ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
        ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
        ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
        ax[0, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
        ax[0, 3].set_title("Silhouette Mask")
        ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
        ax[1, 3].set_title("Loss Mask")
        # Turn off axis
        for i in range(2):
            for j in range(4):
                ax[i, j].axis('off')
        # Set Title
        fig.suptitle(f"Tracking Iteration: {tracking_iteration}", fontsize=16)
        # Figure Tight Layout
        fig.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"tmp.png"), bbox_inches='tight')
        plt.close()
        plot_img = cv2.imread(os.path.join(plot_dir, f"tmp.png"))
        cv2.imshow('Diff Images', plot_img)
        cv2.waitKey(1)
        ## Save Tracking Loss Viz
        # save_plot_dir = os.path.join(plot_dir, f"tracking_%04d" % iter_time_idx)
        # os.makedirs(save_plot_dir, exist_ok=True)
        # plt.savefig(os.path.join(save_plot_dir, f"%04d.png" % tracking_iteration), bbox_inches='tight')
        # plt.close()

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses

def prune_gaussians_w_semantic(params, variables, optimizer, iter, prune_dict, knn=3):
    if iter <= prune_dict['stop_after']:
        if (iter >= prune_dict['start_after']) and (iter % prune_dict['prune_every'] == 0):
            if iter == prune_dict['stop_after']:
                remove_threshold = prune_dict['final_removal_opacity_threshold']
            else:
                remove_threshold = prune_dict['removal_opacity_threshold']
            # Remove Gaussians with low opacity
            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze() #

            # ### replace the low pro labels with nearest lables


            # Remove Gaussians that are too big
            if iter >= prune_dict['remove_big_after']:
                big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.1 * variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)


            params, variables = remove_points(to_remove, params, variables, optimizer)

        # Reset Opacities for all Gaussians
        if iter > 0 and iter % prune_dict['reset_opacities_every'] == 0 and prune_dict['reset_opacities']:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables

def remove_points(to_remove, params, variables, optimizer):
    to_keep = ~to_remove
    keys = [k for k in params.keys() if k not in ['cam_unnorm_rots', 'cam_trans']]
    for k in keys:
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][to_keep]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][to_keep]
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter((group["params"][0][to_keep].requires_grad_(True)))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(group["params"][0][to_keep].requires_grad_(True))
            params[k] = group["params"][0]
    variables['means2D_gradient_accum'] = variables['means2D_gradient_accum'][to_keep]
    variables['denom'] = variables['denom'][to_keep]
    variables['max_2D_radius'] = variables['max_2D_radius'][to_keep]
    if 'seman_cls_ids' in variables.keys():
        variables['seman_cls_ids'] = variables['seman_cls_ids'][to_keep]
    if 'timestep' in variables.keys():
        variables['timestep'] = variables['timestep'][to_keep]
    return params, variables


def densify(params, variables, optimizer, iter, densify_dict):
    if iter <= densify_dict['stop_after']:
        variables = accumulate_mean2d_gradient(variables)
        grad_thresh = densify_dict['grad_thresh']
        if (iter >= densify_dict['start_after']) and (iter % densify_dict['densify_every'] == 0):
            grads = variables['means2D_gradient_accum'] / variables['denom']
            grads[grads.isnan()] = 0.0
            to_clone = torch.logical_and(grads >= grad_thresh, (
                        torch.max(torch.exp(params['log_scales']), dim=1).values <= 0.01 * variables['scene_radius']))
            new_params = {k: v[to_clone] for k, v in params.items() if k not in ['cam_unnorm_rots', 'cam_trans']}
            params = cat_params_to_optimizer(new_params, params, optimizer)
            num_pts = params['means3D'].shape[0]

            padded_grad = torch.zeros(num_pts, device="cuda")
            padded_grad[:grads.shape[0]] = grads
            to_split = torch.logical_and(padded_grad >= grad_thresh,
                                         torch.max(torch.exp(params['log_scales']), dim=1).values > 0.01 * variables[
                                             'scene_radius'])
            n = densify_dict['num_to_split_into']  # number to split into
            new_params = {k: v[to_split].repeat(n, 1) for k, v in params.items() if k not in ['cam_unnorm_rots', 'cam_trans']}
            stds = torch.exp(params['log_scales'])[to_split].repeat(n, 3)
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(params['unnorm_rotations'][to_split]).repeat(n, 1, 1)
            new_params['means3D'] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            new_params['log_scales'] = torch.log(torch.exp(new_params['log_scales']) / (0.8 * n))
            params = cat_params_to_optimizer(new_params, params, optimizer)
            if 'seman_cls_ids' in variables.keys():
                new_variables ={
                    'seman_cls_ids': variables['seman_cls_ids'][to_split].repeat(n, 1),
                }
                variables['seman_cls_ids'] = torch.cat((variables['seman_cls_ids'], new_variables['seman_cls_ids']),dim=0)

            variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda")
            variables['denom'] = torch.zeros(num_pts, device="cuda")
            variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda")
            to_remove = torch.cat((to_split, torch.zeros(n * to_split.sum(), dtype=torch.bool, device="cuda")))
            params, variables = remove_points(to_remove, params, variables, optimizer)

            if iter == densify_dict['stop_after']:
                remove_threshold = densify_dict['final_removal_opacity_threshold']
            else:
                remove_threshold = densify_dict['removal_opacity_threshold']
            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()
            if iter >= densify_dict['remove_big_after']:
                big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.1 * variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)
            params, variables = remove_points(to_remove, params, variables, optimizer)

            # torch.cuda.empty_cache()

        # Reset Opacities for all Gaussians (This is not desired for mapping on only current frame)
        if iter > 0 and iter % densify_dict['reset_opacities_every'] == 0 and densify_dict['reset_opacities']:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


