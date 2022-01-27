import os

import matplotlib.pyplot as plt
import torch

import cost_functions as cf
import rigid_registration as rr
import global_affine_registration as ga
import local_affine_registration as la
import nonrigid_registration as nr
from network import global_affine_network as gan
from network import mm_local_affine_network as lan
from network import mm_nonrigid_network as nrn

import utils
import paths
import Parameter


def registration(id, source_path, target_path, reg_params, iter_params, plot=False, device=torch.device("cpu")):

    print(f"Source: {source_path}")
    print(f"Target: {target_path}")

    # save settings (used for generating the files and figures)
    performed_steps = ""  # currently performed step
    source_str = source_path.split("/")[-1]  # name of the source image
    target_str = target_path.split("/")[-1]  # name of the target image
    output_path = os.path.join(os.path.join(Parameter.get_value("ir_output_path"), id))  # save path of the calculated displacement fields
    preview_path = os.path.join(os.path.join(paths.preview_folder, id))  # save path of the preview figures
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(preview_path):
        os.makedirs(preview_path)

    # similarity metric of the image pair (used for the preview figure)
    cost_func = cf.loss_global
    cost_func_params = dict()
    cost_func_params["mode"] = Parameter.get_value("similarity_metric")

    ### Load the registration params
    rigid_registration = reg_params['rigid_registration']
    global_affine_registration = reg_params['global_affine_registration']
    local_affine_registration = reg_params['local_affine_registration']
    nonrigid_registration = reg_params['nonrigid_registration']

    ### Perform the registration steps
    if rigid_registration:
        # load the rigid registration params
        rr_image_level = Parameter.get_value("rr_image_level")
        angle_step = Parameter.get_value("angle_step")
        # load the image pair
        source = utils.load_whole_image(source_path, rr_image_level, device=device)
        target = utils.load_whole_image(target_path, rr_image_level, device=device)
        # perform the rigid registration
        rr_displacement_field = rr.rigid_registration(source, target, angle_step, device=device)
        curr_displacement_field = rr_displacement_field
        # save the current displacement field
        performed_steps += "rr_"
        torch.save(curr_displacement_field, os.path.join(output_path, f"{performed_steps}displacement_field_level{rr_image_level}.pt"))

        if plot:
            # calculate loss
            new_source = utils.warp_tensors(source, rr_displacement_field, device=device).view(source.size(0), source.size(1))
            loss_before = cost_func(source, target, device=device, **cost_func_params)
            loss_after = cost_func(new_source, target, device=device, **cost_func_params)

            plt.figure(figsize=(10, 8))
            plt.suptitle(f"Source: {source_str}\nTarget: {target_str}\nlevel: {rr_image_level}")
            plt.subplot(2, 3, 1)
            plt.title("Source")
            plt.imshow(source.detach().cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.subplot(2, 3, 2)
            plt.title("Target")
            plt.imshow(target.detach().cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.subplot(2, 3, 3)
            plt.title("Subtraction (NCC = %.4f)" % loss_before.item())
            plt.imshow((target.detach() - source.detach()).cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.subplot(2, 3, 4)
            plt.title("Transformed source")
            plt.imshow(new_source.detach().cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.subplot(2, 3, 5)
            plt.title("Target")
            plt.imshow(target.detach().cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.subplot(2, 3, 6)
            plt.title("Subtraction (NCC = %.4f)" % loss_after.item())
            plt.imshow((target.detach() - new_source.detach()).cpu().numpy(), cmap='gray')
            plt.axis('off')

            plt.savefig(os.path.join(preview_path, f"{performed_steps}{id}.png"))
            plt.close()

    if global_affine_registration:
        # load the global affine registration params
        ga_image_level = Parameter.get_value("ga_image_level")
        ga_model_path = Parameter.get_value("ga_model_path")
        # load the global affine registration network
        ga_model = gan.load_network(device=device, path=ga_model_path)
        # load the image pair
        source = utils.load_whole_image(source_path, ga_image_level, device=device)
        target = utils.load_whole_image(target_path, ga_image_level, device=device)

        warped_source = source.clone()
        # if other registration steps have been performed previously, the original source needs to be warped first
        if rigid_registration:
            # resample the current displacement field to the size of source image
            curr_displacement_field = utils.resample_displacement_fields(curr_displacement_field, (1, 2, source.size(0), source.size(1)), padding_mode='border', device=device)
            # warp the source image
            warped_source = utils.warp_tensors(warped_source, curr_displacement_field, device=device)[0, 0, :, :]
            # perform the global affine registration
            ga_transform = ga.global_affine_registration(warped_source, target, ga_model, device=device)
            # convert the calculated transformation matrix to the displacement field
            ga_displacement_field = utils.transform_to_displacement_fields(source, ga_transform, device=device)[0, :, :, :]  # of size (2, height, width)
            # combine the previous displacement field and the displacement field calculated in this step
            curr_displacement_field = utils.compose_displacement_fields(curr_displacement_field, ga_displacement_field, device=device)
        else:
            # directly perform the global affine registration
            ga_transform = ga.global_affine_registration(warped_source, target, ga_model, device=device)
            # convert the calculated transformation matrix to the displacement field
            ga_displacement_field = utils.transform_to_displacement_fields(source, ga_transform, device=device)[0, :, :, :]  # (2, height, width)
            curr_displacement_field = ga_displacement_field
        # save the current displacement field
        performed_steps += "ga_"
        torch.save(curr_displacement_field, os.path.join(output_path, f"{performed_steps}displacement_field_level{ga_image_level}.pt"))

        if plot:
            # calculate loss
            new_source = utils.warp_tensors(warped_source, ga_displacement_field, device=device).view(source.size(0), source.size(1))
            loss_before = cost_func(warped_source, target, device=device, **cost_func_params)
            loss_after = cost_func(new_source, target, device=device, **cost_func_params)

            plt.figure(figsize=(10, 8))
            plt.suptitle(f"Source: {source_str}\nTarget: {target_str}\nlevel: {ga_image_level}")
            plt.subplot(2, 3, 1)
            plt.title("Source")
            plt.imshow(warped_source.detach().cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.subplot(2, 3, 2)
            plt.title("Target")
            plt.imshow(target.detach().cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.subplot(2, 3, 3)
            plt.title("Subtraction (NCC = %.4f)" % loss_before.item())
            plt.imshow((target.detach() - warped_source.detach()).cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.subplot(2, 3, 4)
            plt.title("Transformed source")
            plt.imshow(new_source.detach().cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.subplot(2, 3, 5)
            plt.title("Target")
            plt.imshow(target.detach().cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.subplot(2, 3, 6)
            plt.title("Subtraction (NCC = %.4f)" % loss_after.item())
            plt.imshow((target.detach() - new_source.detach()).cpu().numpy(), cmap='gray')
            plt.axis('off')

            plt.savefig(os.path.join(preview_path, f"{performed_steps}{id}.png"))
            plt.close()

    if local_affine_registration:
        # load the global affine registration params
        la_inner_iteration = iter_params["la_iteration"]
        la_image_level = Parameter.get_value("la_image_level")
        la_model_path = Parameter.get_value("la_model_path")
        # load the local affine registration network
        la_model = lan.load_network(device=device, path=la_model_path)
        # load the image pair
        source = utils.load_whole_image(source_path, la_image_level, device=device)
        target = utils.load_whole_image(target_path, la_image_level, device=device)
        # perform the local affine registration
        warped_source = source.clone()
        if rigid_registration or global_affine_registration:
            curr_displacement_field = utils.resample_displacement_fields(curr_displacement_field, (1, 2, source.size(0), source.size(1)), padding_mode='border', device=device)
            warped_source = utils.warp_tensors(warped_source, curr_displacement_field, device=device)[0, 0, :, :]
            la_displacement_field = la.local_affine_registration(warped_source, target, la_model, la_inner_iteration, device=device)  # of size (2, height, width)
            curr_displacement_field = utils.compose_displacement_fields(curr_displacement_field, la_displacement_field, device=device)
        else:
            la_displacement_field = la.local_affine_registration(warped_source, target, la_model, la_inner_iteration, device=device)  # of size (2, height, width)
            curr_displacement_field = la_displacement_field
        # save the current displacement field
        performed_steps += "la" + str(la_inner_iteration) + "_"
        torch.save(curr_displacement_field, os.path.join(output_path, f"{performed_steps}displacement_field_level{la_image_level}.pt"))

        if plot:
            # calculate loss
            new_source = utils.warp_tensors(warped_source, la_displacement_field, device=device).view(source.size(0), source.size(1))
            loss_before = cost_func(warped_source, target, device=device, **cost_func_params)
            loss_after = cost_func(new_source, target, device=device, **cost_func_params)

            plt.figure(figsize=(10, 8))
            plt.suptitle(f"Source: {source_str}\nTarget: {target_str}\nlevel: {la_image_level}   Iteration: {la_inner_iteration}")
            plt.subplot(2, 3, 1)
            plt.title("Source")
            plt.imshow(warped_source.detach().cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.subplot(2, 3, 2)
            plt.title("Target")
            plt.imshow(target.detach().cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.subplot(2, 3, 3)
            plt.title("Subtraction (NCC = %.4f)" % loss_before.item())
            plt.imshow((target.detach() - warped_source.detach()).cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.subplot(2, 3, 4)
            plt.title("Transformed source")
            plt.imshow(new_source.detach().cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.subplot(2, 3, 5)
            plt.title("Target")
            plt.imshow(target.detach().cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.subplot(2, 3, 6)
            plt.title("Subtraction (NCC = %.4f)" % loss_after.item())
            plt.imshow((target.detach() - new_source.detach()).cpu().numpy(), cmap='gray')
            plt.axis('off')

            plt.savefig(os.path.join(preview_path, f"{performed_steps}{id}.png"))
            plt.close()

    if nonrigid_registration:
        # load the non-rigid registration params
        nr_inner_iteration = iter_params["nr_iteration"]
        nr_image_level = Parameter.get_value("nr_image_level")
        nr_model_path = Parameter.get_value("nr_model_path")
        # load the non-rigid registration network
        nr_model = nrn.load_network(device=device, path=nr_model_path)
        # load the image pair
        source = utils.load_whole_image(source_path, nr_image_level, device=device)
        target = utils.load_whole_image(target_path, nr_image_level, device=device)
        # perform the non-rigid registration
        warped_source = source.clone()
        if rigid_registration or global_affine_registration or local_affine_registration:
            curr_displacement_field = utils.resample_displacement_fields(curr_displacement_field, (1, 2, source.size(0), source.size(1)), padding_mode='border', device=device)
            warped_source = utils.warp_tensors(warped_source, curr_displacement_field, device=device)[0, 0, :, :]
            nr_displacement_field = nr.nonrigid_registration(warped_source, target, nr_model, nr_inner_iteration, device=device)  # of size (2, height, width)
            curr_displacement_field = utils.compose_displacement_fields(curr_displacement_field, nr_displacement_field, device=device)
        else:
            nr_displacement_field = nr.nonrigid_registration(warped_source, target, nr_model, nr_inner_iteration, device=device)  # of size (2, height, width)
            curr_displacement_field = nr_displacement_field
        # save the current displacement field
        performed_steps += "nr" + str(nr_inner_iteration) + "_"
        torch.save(curr_displacement_field, os.path.join(output_path, f"{performed_steps}displacement_field_level{nr_image_level}.pt"))

        if plot:
            # calculate loss
            new_source = utils.warp_tensors(warped_source, nr_displacement_field, device=device).view(source.size(0), source.size(1))
            loss_before = cost_func(warped_source, target, device=device, **cost_func_params)
            loss_after = cost_func(new_source, target, device=device, **cost_func_params)

            plt.figure(figsize=(10, 8))
            plt.suptitle(f"Source: {source_str}\nTarget: {target_str}\nlevel: {nr_image_level}   Iteration: {nr_inner_iteration}")
            plt.subplot(2, 3, 1)
            plt.title("Source")
            plt.imshow(warped_source.detach().cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.subplot(2, 3, 2)
            plt.title("Target")
            plt.imshow(target.detach().cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.subplot(2, 3, 3)
            plt.title("Subtraction (NCC = %.4f)" % loss_before.item())
            plt.imshow((target.detach() - warped_source.detach()).cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.subplot(2, 3, 4)
            plt.title("Transformed source")
            plt.imshow(new_source.detach().cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.subplot(2, 3, 5)
            plt.title("Target")
            plt.imshow(target.detach().cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.subplot(2, 3, 6)
            plt.title("Subtraction (NCC = %.4f)" % loss_after.item())
            plt.imshow((target.detach() - new_source.detach()).cpu().numpy(), cmap='gray')
            plt.axis('off')

            plt.savefig(os.path.join(preview_path, f"{performed_steps}{id}.png"))
            plt.close()
