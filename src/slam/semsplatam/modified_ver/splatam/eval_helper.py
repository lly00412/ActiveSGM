import os
import cv2
import torch
from tqdm import tqdm
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from imgviz import label_colormap

from third_parties.splatam.datasets.gradslam_datasets.geometryutils import relative_transformation
from third_parties.splatam.utils.slam_external import build_rotation, calc_psnr
from third_parties.splatam.utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
)
from third_parties.splatam.utils.eval_helpers import evaluate_ate
# modified version
from src.slam.splatam.eval_helper import transform_to_frame,resize_tensor
from src.utils.general_utils import *
from src.slam.semsplatam.modified_ver.splatam.splatam import calc_cosine, transformed_params2semrendervar,setup_camera
import matplotlib.pyplot as plt

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from channel_rasterization import GaussianRasterizer as SEMRenderer

loss_fn_alex = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()

def plot_rgbds_silhouette_fast(color, depth, seman_rgb, rastered_color, rastered_depth,rastered_seman_rgb, presence_sil_mask, diff_depth_l1,
                         psnr, depth_l1, fig_title, plot_dir=None, plot_name=None, 
                         save_plot=False, wandb_run=None, wandb_step=None, wandb_title=None, 
                         diff_rgb=None, 
                         target_res=(256, 256), use_nearest_interp=True):        
        
    # Resize images for faster plotting if target_res is provided
    mode = 'nearest' if use_nearest_interp else 'bilinear'
    color = resize_tensor(color, *target_res, mode=mode)
    depth = resize_tensor(depth, *target_res, mode=mode)
    seman_rgb = torch.from_numpy(seman_rgb/255.).permute(2,0,1) # 3, H ,W
    seman = resize_tensor(seman_rgb, *target_res, mode=mode)
    rastered_color = resize_tensor(rastered_color, *target_res, mode=mode)
    rastered_depth = resize_tensor(rastered_depth, *target_res, mode=mode)
    rastered_seman = torch.from_numpy(rastered_seman_rgb / 255.).permute(2,0,1)
    rastered_seman = resize_tensor(rastered_seman, *target_res, mode=mode)

    # Convert tensors to numpy arrays (convert from torch to numpy)
    color_np = color.permute(1, 2, 0).cpu().numpy() * 255  # (H, W, C)
    depth_np = depth[0, :, :].cpu().numpy()          # (H, W)
    rastered_color_np = rastered_color.permute(1, 2, 0).cpu().numpy() * 255  # (H, W, C)
    rastered_depth_np = rastered_depth[0, :, :].cpu().numpy()          # (H, W)
    diff_depth_l1_np = diff_depth_l1.squeeze(0).cpu().numpy()          # (H, W)
    seman_np = seman.permute(1,2,0).cpu().numpy()*255
    rastered_seman_np = rastered_seman.permute(1,2,0).cpu().numpy()*255

    # Scale the depth images for better visualization in grayscale
    depth_np = cv2.applyColorMap(cv2.convertScaleAbs(depth_np, alpha=255 / 6.0), cv2.COLORMAP_JET)
    rastered_depth_np = cv2.applyColorMap(cv2.convertScaleAbs(rastered_depth_np, alpha=255 / 6.0), cv2.COLORMAP_JET)
    diff_depth_l1_np = cv2.applyColorMap(cv2.convertScaleAbs(diff_depth_l1_np, alpha=255 / 6.0), cv2.COLORMAP_JET)
    diff_depth_l1_np = cv2.resize(diff_depth_l1_np, target_res, interpolation=cv2.INTER_NEAREST)

    # Convert Silhouette or diff_rgb to numpy, if applicable
    if diff_rgb is not None:
        diff_rgb_np = diff_rgb.cpu().numpy()  # (H, W)
        diff_rgb_np = cv2.applyColorMap(cv2.convertScaleAbs(diff_rgb_np, alpha=255 / 6.0), cv2.COLORMAP_JET)
    else:
        # Resize presence_sil_mask to match the target resolution
        presence_sil_mask_np = cv2.resize(presence_sil_mask.astype(np.uint8) * 255, target_res, interpolation=cv2.INTER_NEAREST)
        
        # Convert the 2D mask to 3D by repeating the mask across the RGB channels (H, W) -> (H, W, 3)
        presence_sil_mask_np = np.stack([presence_sil_mask_np] * 3, axis=-1)

    # Stack the images horizontally for row 1 and row 2
    row1 = np.hstack([color_np, depth_np, diff_rgb_np if diff_rgb is not None else presence_sil_mask_np, seman_np])
    row2 = np.hstack([rastered_color_np, rastered_depth_np, diff_depth_l1_np, rastered_seman_np]).astype(np.uint8)


    # Concatenate rows vertically to form the final image
    final_image = np.vstack([row1, row2]).astype(np.uint8)

    # Save the image using OpenCV
    if save_plot and plot_dir is not None and plot_name is not None:
        save_path = os.path.join(plot_dir, f"{plot_name}.png")
        cv2.imwrite(save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV


    if wandb_run is not None:
        if wandb_step is None:
            wandb_run.log({wandb_title: final_image})
        else:
            wandb_run.log({wandb_title: final_image}, step=wandb_step)

    # Optionally return the final image for display or logging (for example in wandb)
    return final_image

# Example Usage
# Assuming color, depth, rastered_color, rastered_depth, etc. are given as input tensors
# plot_rgbd_silhouette(color, depth, rastered_color, rastered_depth, presence_sil_mask, diff_depth_l1, psnr, depth_l1, fig



def eval(slam_model, dataset, final_params, final_variables, num_frames, eval_dir, sil_thres,
         mapping_iters, add_new_gaussians, wandb_run=None, wandb_save_qual=False, eval_every=1, save_frames=False,
         ignore_first_frame = False):
    print("Evaluating Final Parameters ...")
    psnr_list = []
    rmse_list = []
    l1_list = []
    lpips_list = []
    ssim_list = []
    cosine_list = []
    plot_dir = os.path.join(eval_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    if save_frames:
        render_rgb_dir = os.path.join(eval_dir, "rendered_rgb")
        os.makedirs(render_rgb_dir, exist_ok=True)
        render_depth_dir = os.path.join(eval_dir, "rendered_depth")
        os.makedirs(render_depth_dir, exist_ok=True)
        rgb_dir = os.path.join(eval_dir, "rgb")
        os.makedirs(rgb_dir, exist_ok=True)
        depth_dir = os.path.join(eval_dir, "depth")
        os.makedirs(depth_dir, exist_ok=True)
        seman_dir = os.path.join(eval_dir, "seman")
        os.makedirs(seman_dir,exist_ok=True)

    gt_w2c_list = []
    num_frames = len(dataset)

    for time_idx in tqdm(range(num_frames)):
         # Get RGB-D Data & Camera Parameters
        color, depth, intrinsics, pose = dataset[time_idx]
        gt_w2c = torch.linalg.inv(pose)
        gt_w2c_list.append(gt_w2c)
        intrinsics = intrinsics[:3, :3]

        seg_img = color.clone().to(slam_model.semantic_device)
        _, seman = slam_model.semantic_annotation(seg_img)

        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        seman = seman.permute(2, 0, 1).to(color.device) # (H, W, C) -> (C, H, W)
        n_cls = seman.shape[0]


        if time_idx == 0:
            # Process Camera Parameters
            first_frame_w2c = torch.linalg.inv(pose)
            # Setup Camera
            cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy(), num_channels=n_cls)
            # sem_colormap = create_class_colormap(seman.shape[0])
            sem_colormap = label_colormap(seman.shape[0])


        # Skip frames if not eval_every
        if time_idx != 0 and (time_idx+1) % eval_every != 0:
            continue

        # Get current frame Gaussians
        transformed_gaussians = transform_to_frame(final_params, time_idx, 
                                                   gaussians_grad=False, 
                                                   camera_grad=False,
                                                   rel_w2c=gt_w2c
                                                   )
 
        # Define current frame data
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'seman':seman, 'id': time_idx, 'intrinsics': intrinsics, 'w2c': first_frame_w2c}
        # Initialize Render Variables
        rendervar = transformed_params2rendervar(final_params, transformed_gaussians)
        depth_sil_rendervar = transformed_params2depthplussilhouette(final_params, curr_data['w2c'],
                                                                     transformed_gaussians)
        # Render Depth & Silhouette
        depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
        rastered_depth = depth_sil[0, :, :].unsqueeze(0)
        # Mask invalid depth in GT
        valid_depth_mask = (curr_data['depth'] > 0)
        rastered_depth_viz = rastered_depth.detach()
        rastered_depth = rastered_depth * valid_depth_mask
        silhouette = depth_sil[1, :, :]
        presence_sil_mask = (silhouette > sil_thres)
        
        # Render RGB and Calculate PSNR
        im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
        seen = radius > 0

        if mapping_iters==0 and not add_new_gaussians:
            weighted_im = im * presence_sil_mask * valid_depth_mask
            weighted_gt_im = curr_data['im'] * presence_sil_mask * valid_depth_mask
        else:
            weighted_im = im * valid_depth_mask
            weighted_gt_im = curr_data['im'] * valid_depth_mask
        psnr = calc_psnr(weighted_im, weighted_gt_im).mean()
        ssim = ms_ssim(weighted_im.unsqueeze(0).cpu(), weighted_gt_im.unsqueeze(0).cpu(), 
                        data_range=1.0, size_average=True)
        lpips_score = loss_fn_alex(torch.clamp(weighted_im.unsqueeze(0), 0.0, 1.0),
                                    torch.clamp(weighted_gt_im.unsqueeze(0), 0.0, 1.0)).item()

        # Render Semantic and compute cosine similarity
        seman_rendervar = transformed_params2semrendervar(final_params, final_variables, transformed_gaussians, seen)
        rastered_seman, _, = SEMRenderer(raster_settings=curr_data['cam'])(**seman_rendervar)  # 133.H.W
        cosine_score = calc_cosine(curr_data['seman'], rastered_seman, dim=0).item()


        gt_cls_id = curr_data['seman'].argmax(0).detach().cpu().numpy()
        gt_seman_rgb = apply_colormap(gt_cls_id, sem_colormap)
        rastered_seman_ids = rastered_seman.argmax(0).detach().cpu().numpy()
        rastered_seman_rgb = apply_colormap(rastered_seman_ids, sem_colormap)

        ### ignore first frame evaluation ###
        if time_idx == 0 and ignore_first_frame:
            continue

        psnr_list.append(psnr.cpu().numpy())
        ssim_list.append(ssim.cpu().numpy())
        lpips_list.append(lpips_score)
        cosine_list.append(cosine_score)

        # Compute Depth RMSE
        if mapping_iters==0 and not add_new_gaussians:
            diff_depth_rmse = torch.sqrt((((rastered_depth - curr_data['depth']) * presence_sil_mask) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - curr_data['depth']) * presence_sil_mask)
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        else:
            diff_depth_rmse = torch.sqrt((((rastered_depth - curr_data['depth'])) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - curr_data['depth']))
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        rmse_list.append(rmse.cpu().numpy())
        l1_list.append(depth_l1.cpu().numpy())

        if save_frames:
            # Save Rendered RGB and Depth
            viz_render_im = torch.clamp(im, 0, 1)
            viz_render_im = viz_render_im.detach().cpu().permute(1, 2, 0).numpy()
            vmin = 0
            vmax = 6
            viz_render_depth = rastered_depth_viz[0].detach().cpu().numpy()
            normalized_depth = np.clip((viz_render_depth - vmin) / (vmax - vmin), 0, 1)
            depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(render_rgb_dir, "gs_{:04d}.png".format(time_idx)), cv2.cvtColor(viz_render_im*255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(render_depth_dir, "gs_{:04d}.png".format(time_idx)), depth_colormap)

            # Save GT RGB and Depth
            viz_gt_im = torch.clamp(curr_data['im'], 0, 1)
            viz_gt_im = viz_gt_im.detach().cpu().permute(1, 2, 0).numpy()
            viz_gt_depth = curr_data['depth'][0].detach().cpu().numpy()
            normalized_depth = np.clip((viz_gt_depth - vmin) / (vmax - vmin), 0, 1)
            depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(rgb_dir, "gt_{:04d}.png".format(time_idx)), cv2.cvtColor(viz_gt_im*255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(depth_dir, "gt_{:04d}.png".format(time_idx)), depth_colormap)

            # Save Semantic RGB
            cv2.imwrite(os.path.join(seman_dir, "gt_{:04d}.png".format(time_idx)),
                        cv2.cvtColor(gt_seman_rgb, cv2.COLOR_RGB2BGR))


        # Plot the Ground Truth and Rasterized RGB & Depth, along with Silhouette
        fig_title = "Time Step: {}".format(time_idx)
        plot_name = "%04d" % time_idx
        presence_sil_mask = presence_sil_mask.detach().cpu().numpy()
        if wandb_run is None:
            plot_rgbds_silhouette_fast(color, depth, gt_seman_rgb, im, rastered_depth_viz, rastered_seman_rgb, presence_sil_mask, diff_depth_l1,
                                 psnr, depth_l1, fig_title, plot_dir, 
                                 plot_name=plot_name, save_plot=True)
        elif wandb_save_qual:
            plot_rgbds_silhouette_fast(color, depth, gt_seman_rgb, im, rastered_depth_viz, rastered_seman_rgb, presence_sil_mask, diff_depth_l1,
                                 psnr, depth_l1, fig_title, plot_dir, 
                                 plot_name=plot_name, save_plot=True,
                                 wandb_run=wandb_run, wandb_step=None, 
                                 wandb_title="Eval/Qual Viz")

    try:
        # Compute the final ATE RMSE
        # Get the final camera trajectory
        num_frames = final_params['cam_unnorm_rots'].shape[-1]
        latest_est_w2c = first_frame_w2c
        latest_est_w2c_list = []
        latest_est_w2c_list.append(latest_est_w2c)
        valid_gt_w2c_list = []
        valid_gt_w2c_list.append(gt_w2c_list[0])
        for idx in range(1, num_frames):
            # Check if gt pose is not nan for this time step
            if torch.isnan(gt_w2c_list[idx]).sum() > 0:
                continue
            interm_cam_rot = F.normalize(final_params['cam_unnorm_rots'][..., idx].detach())
            interm_cam_trans = final_params['cam_trans'][..., idx].detach()
            intermrel_w2c = torch.eye(4).cuda().float()
            intermrel_w2c[:3, :3] = build_rotation(interm_cam_rot)
            intermrel_w2c[:3, 3] = interm_cam_trans
            latest_est_w2c = intermrel_w2c
            latest_est_w2c_list.append(latest_est_w2c)
            valid_gt_w2c_list.append(gt_w2c_list[idx])
        gt_w2c_list = valid_gt_w2c_list
        # Calculate ATE RMSE
        ate_rmse = evaluate_ate(gt_w2c_list, latest_est_w2c_list)
        print("Final Average ATE RMSE: {:.2f} cm".format(ate_rmse*100))
        if wandb_run is not None:
            wandb_run.log({"Final Stats/Avg ATE RMSE": ate_rmse,
                        "Final Stats/step": 1})
    except:
        ate_rmse = 100.0
        print('Failed to evaluate trajectory with alignment.')
    
    # Compute Average Metrics
    psnr_list = np.array(psnr_list)
    rmse_list = np.array(rmse_list)
    l1_list = np.array(l1_list)
    ssim_list = np.array(ssim_list)
    lpips_list = np.array(lpips_list)
    avg_psnr = psnr_list.mean()
    avg_rmse = rmse_list.mean()
    avg_l1 = l1_list.mean()
    avg_ssim = ssim_list.mean()
    avg_lpips = lpips_list.mean()
    print("Average PSNR: {:.2f}".format(avg_psnr))
    print("Average Depth RMSE: {:.2f} cm".format(avg_rmse*100))
    print("Average Depth L1: {:.2f} cm".format(avg_l1*100))
    print("Average MS-SSIM: {:.3f}".format(avg_ssim))
    print("Average LPIPS: {:.3f}".format(avg_lpips))

    if wandb_run is not None:
        wandb_run.log({"Final Stats/Average PSNR": avg_psnr, 
                       "Final Stats/Average Depth RMSE": avg_rmse,
                       "Final Stats/Average Depth L1": avg_l1,
                       "Final Stats/Average MS-SSIM": avg_ssim, 
                       "Final Stats/Average LPIPS": avg_lpips,
                       "Final Stats/step": 1})

    # Save metric lists as text files
    np.savetxt(os.path.join(eval_dir, "psnr.txt"), psnr_list)
    np.savetxt(os.path.join(eval_dir, "rmse.txt"), rmse_list)
    np.savetxt(os.path.join(eval_dir, "l1.txt"), l1_list)
    np.savetxt(os.path.join(eval_dir, "ssim.txt"), ssim_list)
    np.savetxt(os.path.join(eval_dir, "lpips.txt"), lpips_list)
    with open(os.path.join(eval_dir, "render_result.txt"), 'w') as f:
        lines = []
        lines.append(f"psnr: {avg_psnr}\n")
        lines.append(f"ssim: {avg_ssim}\n")
        lines.append(f"lpips: {avg_lpips}\n")
        lines.append(f"l1(cm): {avg_l1*100}\n")
        lines.append(f"rmse: {avg_rmse}\n")
        f.writelines(lines)

    # Plot PSNR & L1 as line plots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(np.arange(len(psnr_list)), psnr_list)
    axs[0].set_title("RGB PSNR")
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("PSNR")
    axs[1].plot(np.arange(len(l1_list)), l1_list*100)
    axs[1].set_title("Depth L1")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("L1 (cm)")
    fig.suptitle("Average PSNR: {:.2f}, Average Depth L1: {:.2f} cm, ATE RMSE: {:.2f} cm".format(avg_psnr, avg_l1*100, ate_rmse*100), y=1.05, fontsize=16)
    plt.savefig(os.path.join(eval_dir, "metrics.png"), bbox_inches='tight')
    if wandb_run is not None:
        wandb_run.log({"Eval/Metrics": fig})
    plt.close()

def report_progress(params, variables, data, i, progress_bar, iter_time_idx, sil_thres, every_i=1, qual_every_i=1,
                    tracking=False, mapping=False, wandb_run=None, wandb_step=None, wandb_save_qual=False, online_time_idx=None,
                    global_logging=True, 
                    eval_dir=None):
    if i % every_i == 0 or i == 1:
        if wandb_run is not None:
            if tracking:
                stage = "Tracking"
            elif mapping:
                stage = "Mapping"
            else:
                stage = "Current Frame Optimization"
        if not global_logging:
            stage = "Per Iteration " + stage

        if tracking:
            # Get list of gt poses
            gt_w2c_list = data['iter_gt_w2c_list']
            valid_gt_w2c_list = []
            
            # Get latest trajectory
            latest_est_w2c = data['w2c']
            latest_est_w2c_list = []
            latest_est_w2c_list.append(latest_est_w2c)
            valid_gt_w2c_list.append(gt_w2c_list[0])
            for idx in range(1, iter_time_idx+1):
                # Check if gt pose is not nan for this time step
                if torch.isnan(gt_w2c_list[idx]).sum() > 0:
                    continue
                interm_cam_rot = F.normalize(params['cam_unnorm_rots'][..., idx].detach())
                interm_cam_trans = params['cam_trans'][..., idx].detach()
                intermrel_w2c = torch.eye(4).cuda().float()
                intermrel_w2c[:3, :3] = build_rotation(interm_cam_rot)
                intermrel_w2c[:3, 3] = interm_cam_trans
                latest_est_w2c = intermrel_w2c
                latest_est_w2c_list.append(latest_est_w2c)
                valid_gt_w2c_list.append(gt_w2c_list[idx])

            # Get latest gt pose
            gt_w2c_list = valid_gt_w2c_list
            iter_gt_w2c = gt_w2c_list[-1]
            # Get euclidean distance error between latest and gt pose
            iter_pt_error = torch.sqrt((latest_est_w2c[0,3] - iter_gt_w2c[0,3])**2 + (latest_est_w2c[1,3] - iter_gt_w2c[1,3])**2 + (latest_est_w2c[2,3] - iter_gt_w2c[2,3])**2)
            if iter_time_idx > 0:
                # Calculate relative pose error
                rel_gt_w2c = relative_transformation(gt_w2c_list[-2], gt_w2c_list[-1])
                rel_est_w2c = relative_transformation(latest_est_w2c_list[-2], latest_est_w2c_list[-1])
                rel_pt_error = torch.sqrt((rel_gt_w2c[0,3] - rel_est_w2c[0,3])**2 + (rel_gt_w2c[1,3] - rel_est_w2c[1,3])**2 + (rel_gt_w2c[2,3] - rel_est_w2c[2,3])**2)
            else:
                rel_pt_error = torch.zeros(1).float()
            
            # Calculate ATE RMSE
            ate_rmse = evaluate_ate(gt_w2c_list, latest_est_w2c_list)
            ate_rmse = np.round(ate_rmse, decimals=6)
            if wandb_run is not None:
                tracking_log = {f"{stage}/Latest Pose Error":iter_pt_error, 
                               f"{stage}/Latest Relative Pose Error":rel_pt_error,
                               f"{stage}/ATE RMSE":ate_rmse}

        # Get current frame Gaussians
        transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                                   gaussians_grad=False,
                                                   camera_grad=False,)

        # Initialize Render Variables
        rendervar = transformed_params2rendervar(params, transformed_gaussians)
        depth_sil_rendervar = transformed_params2depthplussilhouette(params, data['w2c'], 
                                                                     transformed_gaussians)
        depth_sil, _, _, = Renderer(raster_settings=data['cam'])(**depth_sil_rendervar)
        rastered_depth = depth_sil[0, :, :].unsqueeze(0)
        valid_depth_mask = (data['depth'] > 0)
        silhouette = depth_sil[1, :, :]
        presence_sil_mask = (silhouette > sil_thres)

        im, radii, _, = Renderer(raster_settings=data['cam'])(**rendervar)
        seen = radii > 0

        seman_rendervar = transformed_params2semrendervar(params, variables, transformed_gaussians, seen)
        rastered_seman, _, = SEMRenderer(raster_settings=data['cam'])(**seman_rendervar)  # 133.H.W

        sem_colormap = create_class_colormap(data['seman'].shape[0])
        seman_ids = data['seman'].argmax(0).detach().cpu().numpy()
        seman_rgb = apply_colormap(seman_ids, sem_colormap)
        rastered_seman_ids = rastered_seman.argmax(0).detach().cpu().numpy()
        rastered_seman_rgb = apply_colormap(rastered_seman_ids, sem_colormap)

        if tracking:
            psnr = calc_psnr(im * presence_sil_mask, data['im'] * presence_sil_mask).mean()
        else:
            psnr = calc_psnr(im, data['im']).mean()

        if tracking:
            diff_depth_rmse = torch.sqrt((((rastered_depth - data['depth']) * presence_sil_mask) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - data['depth']) * presence_sil_mask)
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        else:
            diff_depth_rmse = torch.sqrt((((rastered_depth - data['depth'])) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - data['depth']))
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()

        if not (tracking or mapping):
            progress_bar.set_postfix({f"Time-Step: {iter_time_idx} | PSNR: {psnr:.{7}} | Depth RMSE: {rmse:.{7}} | L1": f"{depth_l1:.{7}}"})
            progress_bar.update(every_i)
        elif tracking:
            progress_bar.set_postfix({f"Time-Step: {iter_time_idx} | Rel Pose Error: {rel_pt_error.item():.{7}} | Pose Error: {iter_pt_error.item():.{7}} | ATE RMSE": f"{ate_rmse.item():.{7}}"})
            progress_bar.update(every_i)
        elif mapping:
            progress_bar.set_postfix({f"Time-Step: {online_time_idx} | Frame {data['id']} | PSNR: {psnr:.{7}} | Depth RMSE: {rmse:.{7}} | L1": f"{depth_l1:.{7}}"})
            progress_bar.update(every_i)
        
        if wandb_run is not None:
            wandb_log = {f"{stage}/PSNR": psnr,
                         f"{stage}/Depth RMSE": rmse,
                         f"{stage}/Depth L1": depth_l1,
                         f"{stage}/step": wandb_step}
            if tracking:
                wandb_log = {**wandb_log, **tracking_log}
            wandb_run.log(wandb_log)
        
        # Silhouette Mask
        presence_sil_mask = presence_sil_mask.detach().cpu().numpy()
        if wandb_save_qual and (i % qual_every_i == 0 or i == 1):

            # Log plot to wandb
            if not mapping:
                fig_title = f"Time-Step: {iter_time_idx} | Iter: {i} | Frame: {data['id']}"
            else:
                fig_title = f"Time-Step: {online_time_idx} | Iter: {i} | Frame: {data['id']}"
            plot_rgbds_silhouette_fast(data['im'], data['depth'], seman_rgb, im, rastered_depth, rastered_seman_rgb, presence_sil_mask, diff_depth_l1,
                                 psnr, depth_l1, fig_title, wandb_run=wandb_run, wandb_step=wandb_step, 
                                 wandb_title=f"{stage} Qual Viz")
        elif eval_dir is not None:
            plot_name = "%04d" % online_time_idx
            fig_title = "Time Step: {}".format(online_time_idx)
            plot_dir = os.path.join(eval_dir, "plots_progress")
            os.makedirs(plot_dir, exist_ok=True)
            plot_rgbds_silhouette_fast(data['im'], data['depth'], seman_rgb, im, rastered_depth, rastered_seman_rgb, presence_sil_mask, diff_depth_l1,
                                 psnr, depth_l1, fig_title, plot_dir, 
                                 plot_name=plot_name, save_plot=True)

def calc_miou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute mean Intersection over Union (mIoU) between a predicted mask and a target mask.
    Only considers classes present in the target mask.

    Args:
        pred (torch.Tensor): Predicted semantic mask of shape (H, W).
        target (torch.Tensor): Target semantic mask of shape (H, W).

    Returns:
        float: Mean IoU score.
    """
    pred_flat = pred.view(-1)  # (H*W, C)
    target_flat = target.view(-1).to(pred_flat.device)  # (H*W,)

    # Only consider valid pixels (non-zero target)
    valid_mask = (target_flat != 0)

    pred = pred_flat[valid_mask]
    target = target_flat[valid_mask]

    classes = torch.unique(torch.cat((pred, target)))
    classes = classes[classes != 0]
    iou_per_class = []

    for cls in classes:
        pred_cls = (pred == cls)
        true_cls = (target == cls)

        intersection = (pred_cls & true_cls).sum().float()
        union = (pred_cls | true_cls).sum().float()

        if union == 0:
            iou = torch.tensor(float('nan'))  # Class not present in prediction and ground truth
        else:
            iou = intersection / union

        iou_per_class.append(iou)

    iou_per_class = torch.stack(iou_per_class)
    miou = torch.nanmean(iou_per_class).item()
    return miou

def calc_iou_per_classes(pred: torch.Tensor, target: torch.Tensor, target_classes= None):
    """
    Compute mean Intersection over Union (mIoU) between a predicted mask and a target mask.
    Only considers classes present in the target mask.

    Args:
        pred (torch.Tensor): Predicted semantic mask of shape (H, W).
        target (torch.Tensor): Target semantic mask of shape (H, W).

    Returns:
        miou: Mean IoU score.
        classes: return clases id
        iou_per_class: iou per clases

    """
    pred_flat = pred.view(-1)  # (H*W, C)
    target_flat = target.view(-1).to(pred_flat.device)  # (H*W,)

    # Only consider valid pixels (non-zero target)
    valid_mask = (target_flat != 0)

    pred = pred_flat[valid_mask]
    target = target_flat[valid_mask]

    if target_classes is not None:
        classes = target_classes
    else:
        classes = torch.unique(torch.cat((pred, target)))
        classes = classes[classes != 0]
    iou_per_class = []

    for cls in classes:
        pred_cls = (pred == cls)
        true_cls = (target == cls)

        intersection = (pred_cls & true_cls).sum().float()
        union = (pred_cls | true_cls).sum().float()

        if union == 0:
            iou = torch.tensor(float('nan')).to(pred_cls.device)  # Class not present in prediction and ground truth
        else:
            iou = intersection / union

        iou_per_class.append(iou)

    iou_per_class = torch.stack(iou_per_class)
    return iou_per_class

# TODO: preprocess like SGS-SLAM
def post_precess_seg(pred_logits, target_id):
    target_id = target_id.reshape(-1,1)
    candidate_id, _ = torch.unique(target_id, dim=0,return_inverse=True)
    post_logits = pred_logits[...,candidate_id].squeeze()
    closest_indices = torch.argmax(post_logits,dim=-1)
    post_id = candidate_id[closest_indices].squeeze()
    return post_id

def calc_topk_acc(pred_logits: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    Computes the top-k accuracy for semantic segmentation.

    Args:
        pred_logits: Tensor of shape (H, W, num_classes)
        target: Tensor of shape (H, W), ground truth class indices
        topk: Tuple of integers (e.g., (1, 5))

    Returns:
        List of accuracies corresponding to each k in topk.
    """
    assert pred_logits.shape[:2] == target.shape, "Shape mismatch between logits and target"
    H, W, C = pred_logits.shape
    pred_logits_flat = pred_logits.view(-1, C)  # (H*W, C)
    target_flat = target.view(-1).to(pred_logits_flat.device)  # (H*W,)

    # Only consider valid pixels (non-zero target)
    valid_mask = (target_flat != 0)
    if valid_mask.sum() == 0:
        return [0.0 for _ in topk]

    pred_logits_flat_valid = pred_logits_flat[valid_mask]  # reduced size
    target_flat_valid = target_flat[valid_mask]  # reduced size

    res = []
    for maxk in topk:
        _, pred_topk = pred_logits_flat_valid.topk(maxk, dim=1, largest=True, sorted=True)  # (valid_pixels, maxk)
        correct = pred_topk.eq(target_flat_valid.unsqueeze(1))  # (valid_pixels, maxk)

        correct_k = correct[:, :maxk].any(dim=1).float().sum()
        acc_k = correct_k / target_flat_valid.numel()
        res.append(acc_k.item())

    return res


def calc_mAP(pred_logits, target):
    """
    Compute mean Average Precision (mAP) for semantic segmentation.

    Args:
        pred_logits (Tensor): shape (H, W, C), per-pixel class scores (logits or probabilities).
        target (Tensor): shape (H, W), per-pixel ground truth class indices.
        num_classes (int, optional): Number of classes. If None, inferred from pred.shape[2].

    Returns:
        mAP (float): mean Average Precision over all classes.
    """
    H, W, C = pred_logits.shape
    num_classes = C

    pred_logits_flat = pred_logits.view(-1, C)  # (H*W, C)
    target_flat = target.view(-1)  # (H*W,)
    valid_mask = (target_flat != 0)

    pred_logits_flat_valid = pred_logits_flat[valid_mask]  # reduced size
    target_flat_valid = target_flat[valid_mask]  # reduced size

    average_precisions = []

    for cls in range(num_classes):
        # Binary ground truth: 1 if pixel belongs to class `cls`, else 0
        true_binary = (target_flat_valid == cls).float()
        scores = pred_logits_flat_valid[:, cls]

        # Sort by predicted score
        sorted_indices = torch.argsort(scores, descending=True)
        true_sorted = true_binary[sorted_indices]

        # Compute precision at each threshold
        cum_true = torch.cumsum(true_sorted, dim=0)
        precision = cum_true / (torch.arange(1, len(true_sorted) + 1, device=pred_logits.device))

        total_positives = true_binary.sum()
        if total_positives == 0:
            ap = torch.tensor(float('nan')).to(pred_logits.device)   # Ignore if class not present in GT
        else:
            ap = (precision * true_sorted).sum() / total_positives

        average_precisions.append(ap)

    average_precisions = torch.stack(average_precisions)
    mAP = torch.nanmean(average_precisions).item()

    return mAP


def calc_f1(pred: torch.Tensor, target: torch.Tensor, eps=1e-7) -> float:
    """
    Compute mean F1-score per class for semantic segmentation.

    Args:
        pred_logits (torch.Tensor): shape (H, W, N_CLASS)
        target (torch.Tensor): shape (H, W)
        eps (float): epsilon for numerical stability

    Returns:
        mean_f1 (float): mean F1-score across classes
        f1_per_class (torch.Tensor): F1-score per class
    """
    pred_flat = pred.view(-1)  # (H*W, C)
    target_flat = target.view(-1).to(pred_flat.device)  # (H*W,)

    # Only consider valid pixels (non-zero target)
    valid_mask = (target_flat != 0)

    pred = pred_flat[valid_mask]
    target = target_flat[valid_mask]

    classes = torch.unique(torch.cat((pred, target)))
    classes = classes[classes != 0]
    f1_per_class = []

    # Compute F1-score per class
    for cls in classes:
        if cls == 0:
            continue  # optionally skip background
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        tp = (pred_cls & target_cls).sum().float()
        fp = (pred_cls & ~target_cls).sum().float()
        fn = (~pred_cls & target_cls).sum().float()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)

        f1 = 2 * precision * recall / (precision + recall + eps)
        f1_per_class.append(f1)

    # Compute mean F1-score across classes excluding class 0
    average_f1s = torch.stack(f1_per_class)
    mean_f1 = torch.nanmean(average_f1s).item()

    return mean_f1


# TODO： add miou and topk acc
@torch.no_grad()
def eval_semantic(slam_model, dataset, final_params, final_variables, num_frames, eval_dir, sil_thres,
         mapping_iters, add_new_gaussians, wandb_run=None, wandb_save_qual=False, eval_every=1, save_frames=False,
         ignore_first_frame=False):
    print("Evaluating Final Parameters ...")
    miou_g_curr_list = []
    miou_p_curr_list = []
    miou_g_list = []
    miou_p_list = []
    top1_g_list = []
    top3_g_list = []
    top5_g_list = []
    top1_p_list = []
    top3_p_list = []
    top5_p_list = []
    mAP_g_list = []
    mAP_p_list = []
    f1_g_list = []
    f1_p_list = []
    plot_dir = os.path.join(eval_dir, "semantic_plots")
    os.makedirs(plot_dir, exist_ok=True)
    if save_frames:
        seman_dir = os.path.join(eval_dir, "seman")
        os.makedirs(seman_dir, exist_ok=True)

    gt_w2c_list = []
    num_frames = len(dataset)

    for time_idx in tqdm(range(num_frames)):
        # Get RGB-D Data & Camera Parameters
        color, _, intrinsics, pose = dataset[time_idx]
        gt_w2c = torch.linalg.inv(pose)
        gt_w2c_list.append(gt_w2c)
        intrinsics = intrinsics[:3, :3]

        seman_gt = dataset.get_semantic_map(time_idx)
        seg_img = color.clone().to(slam_model.semantic_device)
        seman_pseudo, seman_pseudo_logits = slam_model.semantic_annotation(seg_img)
        seman_pseudo = seman_pseudo.to(slam_model.device)
        seman_pseudo_logits = seman_pseudo_logits.to(slam_model.device)
        n_cls = seman_pseudo_logits.shape[-1]

        if time_idx == 0:
            # Process Camera Parameters
            first_frame_w2c = torch.linalg.inv(pose)
            # Setup Camera
            cam = setup_camera(color.shape[1], color.shape[0], intrinsics.cpu().numpy(),
                               first_frame_w2c.detach().cpu().numpy(), num_channels=n_cls)

            sem_colormap = label_colormap(n_cls)


        # Skip frames if not eval_every
        if time_idx != 0 and (time_idx + 1) % eval_every != 0:
            continue

        # Get current frame Gaussians
        # transformed_gaussians = transform_to_frame(final_params, time_idx,
        #                                            gaussians_grad=False,
        #                                            camera_grad=False)
        transformed_gaussians = transform_to_frame(final_params, time_idx,
                                                   gaussians_grad=False,
                                                   camera_grad=False,
                                                   rel_w2c=gt_w2c
                                                   )

        # Define current frame data
        curr_data = {'cam': cam, 'im': color, 'seman_gt': seman_gt[0],
                     'seman_pseudo':seman_pseudo, 'seman_pseudo_logits':seman_pseudo_logits,
                     'id': time_idx, 'intrinsics': intrinsics,
                     'w2c': first_frame_w2c}
        # Initialize Render Variables
        rendervar = transformed_params2rendervar(final_params, transformed_gaussians)
        # Render RGB and Calculate PSNR
        _, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
        seen = radius > 0

        # Render Semantic and compute cosine similarity
        seman_rendervar = transformed_params2semrendervar(final_params, final_variables, transformed_gaussians, seen)
        rastered_seman, _, = SEMRenderer(raster_settings=curr_data['cam'])(**seman_rendervar)  # 133.H.W

        rastered_seman = torch.nan_to_num(rastered_seman, nan=0.0)
        rastered_seman[rastered_seman<0] = 0.0

        ## compute semantic metrics
        rastered_seman = rastered_seman.permute(1,2,0)
        rastered_cls_ids = rastered_seman.argmax(-1)

        topks_g = calc_topk_acc(pred_logits=rastered_seman, target=curr_data['seman_gt'].long(), topk=(1 ,3, 5))
        topks_p = calc_topk_acc(pred_logits=curr_data['seman_pseudo_logits'], target=curr_data['seman_gt'].long(), topk=(1, 3, 5))
        top1_g_list.append(topks_g[0])
        top3_g_list.append(topks_g[1])
        top5_g_list.append(topks_g[2])
        top1_p_list.append(topks_p[0])
        top3_p_list.append(topks_p[1])
        top5_p_list.append(topks_p[2])

        mAP_g = calc_mAP(pred_logits=rastered_seman,target=curr_data['seman_gt'].long())
        mAP_p = calc_mAP(pred_logits=curr_data['seman_pseudo_logits'],target=curr_data['seman_gt'].long())
        mAP_g_list.append(mAP_g)
        mAP_p_list.append(mAP_p)

        miou_g = calc_miou(pred=rastered_cls_ids,target=curr_data['seman_gt'].long())
        miou_p = calc_miou(pred=curr_data['seman_pseudo'].long(),target=curr_data['seman_gt'].long())
        miou_g_list.append(miou_g)
        miou_p_list.append(miou_p)

        f1_g = calc_f1(pred=rastered_cls_ids,target=curr_data['seman_gt'].long())
        f1_p = calc_f1(pred=curr_data['seman_pseudo'].long(),target=curr_data['seman_gt'].long())
        f1_g_list.append(f1_g)
        f1_p_list.append(f1_p)

        # recolor seg
        reprocess_ids = post_precess_seg(rastered_seman.clone(), curr_data['seman_gt'].long())
        reprocess_pseudo_ids = post_precess_seg(curr_data['seman_pseudo_logits'].clone(), curr_data['seman_gt'].long())
        miou_g_curr = calc_miou(pred=reprocess_ids, target=curr_data['seman_gt'].long())
        miou_p_curr = calc_miou(pred=reprocess_pseudo_ids, target=curr_data['seman_gt'].long())
        miou_g_curr_list.append(miou_g_curr)
        miou_p_curr_list.append(miou_p_curr)


        gt_cls_id = curr_data['seman_gt'].detach().cpu().long().numpy()
        pseudo_cls_id = curr_data['seman_pseudo'].detach().cpu().long().numpy()
        gt_seman_rgb = apply_colormap(gt_cls_id, sem_colormap)
        pseudo_seman_rgb = apply_colormap(pseudo_cls_id, sem_colormap)
        rastered_cls_ids = rastered_cls_ids.detach().cpu().long().numpy()
        rastered_seman_rgb = apply_colormap(rastered_cls_ids, sem_colormap)

        reprocess_pseudo_ids = reprocess_pseudo_ids.detach().cpu().long().numpy()
        recolored_pseudo_seman_rgb = apply_colormap(reprocess_pseudo_ids, sem_colormap)
        reprocess_ids = reprocess_ids.detach().cpu().long().numpy()
        recolored_rastered_seman_rgb = apply_colormap(reprocess_ids, sem_colormap)

        # Plot the Ground Truth, Pseudo and Rasterized semantic RGB
        fig_title = "Time Step: {}".format(time_idx)
        plot_name = "%04d" % time_idx


        #save original one
        save_path = os.path.join(plot_dir, f"{plot_name}_gt.png")
        cv2.imwrite(save_path, cv2.cvtColor(gt_seman_rgb, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

        save_path = os.path.join(plot_dir, f"{plot_name}_render_miou_{miou_g_curr:.4f}.png")
        cv2.imwrite(save_path, cv2.cvtColor(recolored_rastered_seman_rgb, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

        rastered_seman = rastered_seman.permute(1, 2, 0)
        prob_logits = rastered_seman[...,76]
        prob_logits = prob_logits.detach().cpu().numpy()
        heatmap_uint8 = np.uint8(255 * prob_logits)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        save_path = os.path.join(plot_dir, f"{plot_name}_prob_sofa.png")
        cv2.imwrite(save_path, colored_heatmap)

        target_res = (256, 256)
        mode = 'nearest'
        gt_seman_rgb = torch.from_numpy(gt_seman_rgb / 255.).permute(2, 0, 1)  # 3, H ,W
        gt_seman = resize_tensor(gt_seman_rgb, *target_res, mode=mode)
        gt_seman_np = gt_seman.permute(1, 2, 0).cpu().numpy() * 255


        pseudo_seman_rgb = torch.from_numpy(pseudo_seman_rgb / 255.).permute(2, 0, 1)  # 3, H ,W
        pseudo_seman = resize_tensor(pseudo_seman_rgb, *target_res, mode=mode)
        pseudo_seman_np = pseudo_seman.permute(1, 2, 0).cpu().numpy() * 255

        recolored_pseudo_seman_rgb = torch.from_numpy(recolored_pseudo_seman_rgb / 255.).permute(2, 0, 1)  # 3, H ,W
        recolored_pseudo_seman = resize_tensor(recolored_pseudo_seman_rgb, *target_res, mode=mode)
        recolored_pseudo_seman_np = recolored_pseudo_seman.permute(1, 2, 0).cpu().numpy() * 255

        rastered_seman_rgb = torch.from_numpy(rastered_seman_rgb / 255.).permute(2, 0, 1)  # 3, H ,W
        rastered_seman = resize_tensor(rastered_seman_rgb, *target_res, mode=mode)
        rastered_seman_np = rastered_seman.permute(1, 2, 0).cpu().numpy() * 255

        recolored_rastered_seman_rgb = torch.from_numpy(recolored_rastered_seman_rgb / 255.).permute(2, 0, 1)  # 3, H ,W
        recolored_rastered_seman = resize_tensor(recolored_rastered_seman_rgb, *target_res, mode=mode)
        recolored_rastered_seman_np = recolored_rastered_seman.permute(1, 2, 0).cpu().numpy() * 255

        final_image = np.hstack([gt_seman_np,pseudo_seman_np,recolored_pseudo_seman_np,rastered_seman_np, recolored_rastered_seman_np]).astype(np.uint8)
        save_path = os.path.join(plot_dir, f"{plot_name}.png")
        cv2.imwrite(save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

    #
    # Compute Average Metrics
    miou_g_curr_list = np.array(miou_g_curr_list)
    miou_p_curr_list = np.array(miou_p_curr_list)
    miou_g_list = np.array(miou_g_list)
    miou_p_list = np.array(miou_p_list)
    top1_g_list = np.array(top1_g_list)
    top3_g_list = np.array(top3_g_list)
    top5_g_list = np.array(top5_g_list)
    top1_p_list = np.array(top1_p_list)
    top3_p_list = np.array(top3_p_list)
    top5_p_list = np.array(top5_p_list)
    mAP_g_list = np.array(mAP_g_list)
    mAP_p_list = np.array(mAP_p_list)
    f1_g_list = np.array(f1_g_list)
    f1_p_list = np.array(f1_p_list)

    avg_miou_g_curr = miou_g_curr_list.mean()
    avg_miou_p_curr = miou_p_curr_list.mean()
    avg_miou_g = miou_g_list.mean()
    avg_miou_p = miou_p_list.mean()
    avg_top1_g = top1_g_list.mean()
    avg_top3_g = top3_g_list.mean()
    avg_top5_g = top5_g_list.mean()
    avg_top1_p = top1_p_list.mean()
    avg_top3_p = top3_p_list.mean()
    avg_top5_p = top5_p_list.mean()
    avg_mAP_g = mAP_g_list.mean()
    avg_mAP_p = mAP_p_list.mean()
    avg_f1_g = f1_g_list.mean()
    avg_f1_p = f1_p_list.mean()

    print("Average MIOU with GT: {:.2f}".format(avg_miou_g * 100))
    print("Average MIOU with GT (current): {:.2f}".format(avg_miou_g_curr * 100))
    print("Average top1 acc with GT: {:.2f}".format(avg_top1_g * 100))
    print("Average top3 acc with GT: {:.2f}".format(avg_top3_g * 100))
    print("Average top5 acc with GT: {:.2f}".format(avg_top5_g * 100))
    print("Average top5 acc with GT: {:.2f}".format(avg_top5_g * 100))
    print("Average mAP with GT: {:.2f}".format(avg_mAP_g * 100))
    print("Average F1 with GT: {:.2f}".format(avg_f1_g * 100))

    print("Average MIOU with Pseudo: {:.2f}".format(avg_miou_p * 100))
    print("Average MIOU with Pseudo (current): {:.2f}".format(avg_miou_p_curr * 100))
    print("Average top1 acc with Pseudo: {:.2f}".format(avg_top1_p * 100))
    print("Average top3 acc with Pseudo: {:.2f}".format(avg_top3_p * 100))
    print("Average top5 acc with Pseudo: {:.2f}".format(avg_top5_p * 100))
    print("Average mAP with Pseudo: {:.2f}".format(avg_mAP_p * 100))
    print("Average F1 with Pseudo: {:.2f}".format(avg_f1_p * 100))


    # Save metric lists as text files
    with open(os.path.join(eval_dir, "semantic_result.txt"), 'w') as f:
        lines = []
        lines.append(f"miou_g: {avg_miou_g * 100}\n")
        lines.append(f"miou_g_curr: {avg_miou_g_curr * 100}\n")
        lines.append(f"top1_g: {avg_top1_g * 100}\n")
        lines.append(f"top3_g: {avg_top3_g * 100}\n")
        lines.append(f"top5_g: {avg_top5_g * 100}\n")
        lines.append(f"mAP_g: {avg_mAP_g * 100}\n")
        lines.append(f"f1_g: {avg_f1_g * 100}\n")

        lines.append(f"miou_p: {avg_miou_p * 100}\n")
        lines.append(f"miou_p_curr: {avg_miou_p_curr * 100}\n")
        lines.append(f"top1_p: {avg_top1_p * 100}\n")
        lines.append(f"top3_p: {avg_top3_p * 100}\n")
        lines.append(f"top5_p: {avg_top5_p * 100}\n")
        lines.append(f"mAP_p: {avg_mAP_p * 100}\n")
        lines.append(f"f1_p: {avg_f1_p * 100}\n")
        f.writelines(lines)

@torch.no_grad()
def eval_semantic_mp3d(slam_model, dataset, final_params, final_variables, num_frames, eval_dir, sil_thres,
                  mapping_iters, add_new_gaussians, wandb_run=None, wandb_save_qual=False, eval_every=1,
                  save_frames=False,
                  ignore_first_frame=False):
    print("Evaluating for MP3D 8 Semantic Classes ...")
    iou_g_list = []
    iou_p_list = []

    plot_dir = os.path.join(eval_dir, "semantic_plots")
    os.makedirs(plot_dir, exist_ok=True)
    if save_frames:
        seman_dir = os.path.join(eval_dir, "seman")
        os.makedirs(seman_dir, exist_ok=True)

    gt_w2c_list = []
    num_frames = len(dataset)

    target_classes = [17, 37, 15, 19, 14, 26, 5, 38]
    target_classes_name = ['ceiling', 'appliances', 'sink', 'stool', 'plant', 'counter', 'table', 'clothes']

    for time_idx in tqdm(range(num_frames)):
        # Get RGB-D Data & Camera Parameters
        color, _, intrinsics, pose = dataset[time_idx]
        gt_w2c = torch.linalg.inv(pose)
        gt_w2c_list.append(gt_w2c)
        intrinsics = intrinsics[:3, :3]

        seman_gt = dataset.get_semantic_map(time_idx)
        seg_img = color.clone().to(slam_model.semantic_device)
        seman_pseudo, seman_pseudo_logits = slam_model.semantic_annotation(seg_img)
        seman_pseudo = seman_pseudo.to(slam_model.device)
        seman_pseudo_logits = seman_pseudo_logits.to(slam_model.device)
        n_cls = seman_pseudo_logits.shape[-1]

        if time_idx == 0:
            # Process Camera Parameters
            first_frame_w2c = torch.linalg.inv(pose)
            # Setup Camera
            cam = setup_camera(color.shape[1], color.shape[0], intrinsics.cpu().numpy(),
                               first_frame_w2c.detach().cpu().numpy(), num_channels=n_cls)

            sem_colormap = label_colormap(n_cls)

        # Skip frames if not eval_every
        if time_idx != 0 and (time_idx + 1) % eval_every != 0:
            continue

        # Get current frame Gaussians
        transformed_gaussians = transform_to_frame(final_params, time_idx,
                                                   gaussians_grad=False,
                                                   camera_grad=False,
                                                   rel_w2c=gt_w2c
                                                   )

        # Define current frame data
        curr_data = {'cam': cam, 'im': color, 'seman_gt': seman_gt[0],
                     'seman_pseudo': seman_pseudo, 'seman_pseudo_logits': seman_pseudo_logits,
                     'id': time_idx, 'intrinsics': intrinsics,
                     'w2c': first_frame_w2c}
        # Initialize Render Variables
        rendervar = transformed_params2rendervar(final_params, transformed_gaussians)
        # Render RGB and Calculate PSNR
        _, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
        seen = radius > 0

        # Render Semantic and compute cosine similarity
        seman_rendervar = transformed_params2semrendervar(final_params, final_variables, transformed_gaussians,
                                                          seen)
        rastered_seman, _, = SEMRenderer(raster_settings=curr_data['cam'])(**seman_rendervar)  # 133.H.W

        rastered_seman = torch.nan_to_num(rastered_seman, nan=0.0)
        rastered_seman[rastered_seman < 0] = 0.0

        ## compute semantic metrics
        rastered_seman = rastered_seman.permute(1, 2, 0)
        rastered_cls_ids = rastered_seman.argmax(-1)

        iou_per_class_g = calc_iou_per_classes(pred=rastered_cls_ids, target=curr_data['seman_gt'].long(), target_classes=target_classes)
        iou_per_class_p = calc_iou_per_classes(pred=curr_data['seman_pseudo'].long(),
                                                                      target=curr_data['seman_gt'].long(),
                                                                      target_classes=target_classes)
        iou_g_list.append(iou_per_class_g)
        iou_p_list.append(iou_per_class_p)

    iou_g_list = torch.stack(iou_g_list)
    iou_p_list = torch.stack(iou_p_list)

    avg_iou_g_per_class = torch.nanmean(iou_g_list,dim=0)
    avg_iou_p_per_class = torch.nanmean(iou_p_list,dim=0)

    for i in range(len(target_classes)):
        print(f"Ours: IoU for {target_classes_name[i]}: {avg_iou_g_per_class[i].item()*100:.2f}")

    for i in range(len(target_classes)):
        print(f"Pseudo: IoU for {target_classes_name[i]}: {avg_iou_p_per_class[i].item()*100:.2f}")

    # # Save metric lists as text files
    with open(os.path.join(eval_dir, "IoU_per_classes.txt"), 'w') as f:
        lines = []
        for i in range(len(target_classes)):
            lines.append(f"iou_g {target_classes_name[i]}: {avg_iou_g_per_class[i].item()* 100}\n")

        for i in range(len(target_classes)):
            lines.append(f"iou_p {target_classes_name[i]}: {avg_iou_p_per_class[i].item()* 100}\n")

        f.writelines(lines)
