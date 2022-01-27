import os

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch
import torch.utils
import torch.nn as nn

from network import segmentation_network as sn
import utils
import paths
import Parameter


def background_segmentation(id, source_path, target_path, levels, save=False, plot=False, device=torch.device("cpu")):
    """
    Filter out background areas from the input image pair using a pre-trained segmentation network
    """
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")

    # load the segmentation params
    bs_image_level = Parameter.get_value("bs_image_level")  # image level used for background segmentation
    bs_model_path = Parameter.get_value("bs_model_path")

    # load the background segmentation network
    bs_model = sn.load_network(device=device, path=bs_model_path)

    # load the image pair
    source = utils.load_whole_image(source_path, bs_image_level, device=device)
    target = utils.load_whole_image(target_path, bs_image_level, device=device)

    # predict the background mask using the pre-trained background segmentation network
    source_mask, target_mask = segment(source, target, bs_model, device=device)

    if plot:
        orig_source = source.clone()
        orig_target = target.clone()

        # filter out the background areas
        source[(source_mask > 0.5) == 0] = 0
        target[(target_mask > 0.5) == 0] = 0

        fig_path = os.path.join(os.path.join(paths.preview_folder, id))
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        # preview of the source
        plt.figure()
        source_str = source_path.split("/")[-1]
        plt.suptitle(f"{source_str} - level{bs_image_level}")
        plt.subplot(1, 3, 1)
        plt.imshow(orig_source.detach().cpu().numpy(), cmap='gray')
        plt.xlabel("original image")
        plt.subplot(1, 3, 2)
        plt.imshow(source_mask.detach().cpu().numpy(), cmap='gray')
        plt.xlabel("background mask")
        plt.subplot(1, 3, 3)
        plt.imshow(source.detach().cpu().numpy(), cmap='gray')
        plt.xlabel("generated image")
        plt.savefig(os.path.join(fig_path, f"bs_{source_str}.png"))

        # preview of the target
        plt.figure()
        target_str = target_path.split("/")[-1]
        plt.suptitle(f"{target_str} - level{bs_image_level}")
        plt.subplot(1, 3, 1)
        plt.imshow(orig_target.detach().cpu().numpy(), cmap='gray')
        plt.xlabel("original image")
        plt.subplot(1, 3, 2)
        plt.imshow(target_mask.detach().cpu().numpy(), cmap='gray')
        plt.xlabel("background mask")
        plt.subplot(1, 3, 3)
        plt.imshow(target.detach().cpu().numpy(), cmap='gray')
        plt.xlabel("generated image")
        plt.savefig(os.path.join(fig_path, f"bs_{target_str}.png"))

        print("The preview of background segmentation has been saved.")

    if save:
        for level in levels:
            print(f"Generating the level-{level} image...")
            generate_filtered_image(source_path, source_mask, (bs_image_level, level), device=device)
            generate_filtered_image(target_path, target_mask, (bs_image_level, level), device=device)

        print("The background segmentration step is done.")


def segment(source, target, model, device=torch.device("cpu")):
    """
    Predict the background regions of the source and target images
    Code from: https://github.com/MWod/DeepHistReg (a bit modified)
    """
    with torch.set_grad_enabled(False):
        output_min_size = 512

        # resample the input image pair to a given size
        new_shape = utils.calculate_new_shape((source.size(0), source.size(1)), output_min_size, mode="min")
        resampled_source = utils.resample_tensor(source, new_shape, device=device)
        resampled_target = utils.resample_tensor(target, new_shape, device=device)

        # predict the background masks of image pair using a deep segmentation network
        source_mask = model(resampled_source.view(1, 1, resampled_source.size(0), resampled_source.size(1)))[0, 0, :, :]
        target_mask = model(resampled_target.view(1, 1, resampled_target.size(0), resampled_target.size(1)))[0, 0, :, :]

        # resample the masks to the original size of input image pair
        source_mask = utils.resample_tensor(source_mask, (source.size(0), source.size(1)), device=device)
        target_mask = utils.resample_tensor(target_mask, (target.size(0), target.size(1)), device=device)

    return source_mask, target_mask


def generate_filtered_image(input_path, background_mask, level, device=torch.device("cpu")):
    """
    Generate and save the image after filtering out the background
    :param input_path: str, path of the image
    :param background_mask: tensor, predicted background mask of input image
    :param level: tuple -- (source level, target level)
        source level: int, the level of input image
        target level: int, the level of image to generate and save
    """
    source_level = level[0]
    target_level = level[1]

    max_patch_size = Parameter.get_value("max_patch_size")
    output_path = Parameter.get_value("bs_output_path")
    path = os.path.join(input_path.replace("/".join(input_path.split("/")[:3]), output_path), f"level_{target_level}")
    if not os.path.exists(path):
        os.makedirs(path)

    if source_level > target_level:
        size = int(max_patch_size / np.power(2, source_level - target_level))
        resample_factor = np.power(2, source_level - target_level)
    else:
        size = int(max_patch_size * np.power(2, target_level - source_level))
        resample_factor = 1 / np.power(2, target_level - source_level)

    n_row = int(np.ceil(background_mask.size(0) / size))
    n_col = int(np.ceil(background_mask.size(1) / size))
    if n_row == 1 or n_col == 1:
        n_row, n_col = 1, 1

    # if up-sampling, pad the original background_mask
    pad_size = 0
    if source_level > target_level:
        background_mask = background_mask.view(1, 1, background_mask.size(0), background_mask.size(1))
        pad_size = 10
        pad = nn.ReplicationPad2d(padding=(pad_size, pad_size, pad_size, pad_size))
        background_mask = pad(background_mask)
        background_mask = background_mask.view(background_mask.size(2), background_mask.size(3))

    # partition resampling operation
    for row in range(n_row):
        for col in range(n_col):
            if row == n_row - 1 and col == n_col - 1:
                temp_mask = background_mask[row*size:, col*size:]
            elif row == n_row - 1:
                temp_mask = background_mask[row*size:, col*size:(col+1)*size+2*pad_size]
            elif col == n_col - 1:
                temp_mask = background_mask[row*size:(row+1)*size+2*pad_size, col*size:]
            else:
                temp_mask = background_mask[row*size:(row+1)*size+2*pad_size, col*size:(col+1)*size+2*pad_size]

            # load the corresponding image/image patch
            temp_img_patch = utils.load_image_patch(
                os.path.join(input_path, f"level_{level[1]}"), (row, col), device=device
            )

            # resample the current part of background mask
            if n_row == 1 and n_col == 1:
                resampled_mask = utils.resample_tensor(
                    temp_mask, (temp_img_patch.size(0), temp_img_patch.size(1)), padding_mode='border', device=device
                )
            else:
                resampled_mask = utils.resample_tensor(
                    temp_mask, (resample_factor * temp_mask.size(0), resample_factor * temp_mask.size(1)), padding_mode='border', device=device
                )

            if pad_size != 0:
                temp = pad_size * resample_factor
                resampled_mask = resampled_mask[temp:-temp, temp:-temp]

            # generate and save new image/image patch
            temp_img_patch[(resampled_mask > 0.5) == 0] = 0

            to_save_mha = sitk.GetImageFromArray(temp_img_patch.cpu().numpy())
            to_save_jpg = sitk.GetImageFromArray((temp_img_patch.cpu().numpy() * 255).astype(np.ubyte))
            to_save_mha_path = os.path.join(path, f"{row}_{col}.mha")
            to_save_jpg_path = os.path.join(path, f"{row}_{col}.jpg")
            sitk.WriteImage(to_save_mha, to_save_mha_path)
            sitk.WriteImage(to_save_jpg, to_save_jpg_path)





