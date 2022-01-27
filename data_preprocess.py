"""
Follow the following process to process all images in turn:
- automatic detect boundary/manual define boundary (level 8)
- Load the images/image patches within the boundary (level 0-8)
- Convert the images/image patches: RGBA -> RGB -> Grayscale
- Save the converted images/image patches
"""

import os
import time
from external import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.color as color
import openslide
import SimpleITK as sitk

import utils
import paths
import Parameter


# General setting
register_levels = Parameter.get_value("register_levels")
max_patch_size = Parameter.get_value("max_patch_size")


def pre_processing(filename, levels, manual_crop=False, manual_boundary=None, format=[], show=False):
    """
    Pre-processing the raw whole slide image, including manual cropping, spliting into image patches and grayscaling.
    The original image will be saved as image patches in both RGB (for generating the final result) and grayscale formats (for registration).

    :param filename: name of whole slide image, str
    :param levels: image levels of the generated results (deformed images), list of int
    :param manual_crop: switch of manual cropping, bool; if manual_crop=False, then performs automatic cropping
    :param manual_boundary: manually determined image boundary, tuple -- (x, y, width, height)
    :param format: image saving format, list of string, e.g., ['mha', 'jpg']
    """

    all_levels = list(set(levels + register_levels))

    for extern_file in [file for file in os.listdir(paths.raw_data_path) if file != ".DS_Store"]:
        for inter_file in os.listdir(os.path.join(paths.raw_data_path, extern_file)):
            if filename + ".mrxs" == inter_file:
                b_t = time.time()

                img_path = os.path.join(paths.raw_data_path, extern_file, inter_file)
                print("Current WSI: ", img_path)

                # open the whole slide image
                slide = openslide.OpenSlide(filename=img_path)

                # load the lowest resolution image
                curr_level = slide.level_count - 1
                curr_level_img = np.array(slide.read_region(location=(0, 0), level=curr_level, size=slide.level_dimensions[curr_level]))

                # detect image boundary
                if not manual_crop:
                    # automatic detect the boundary of image
                    x, y, w, h = utils.boundary_detect(curr_level_img)
                else:
                    # manually determined boundary of image
                    assert manual_boundary is not None
                    x, y, w, h = manual_boundary

                downsample_factor = int(slide.level_downsamples[curr_level] / slide.level_downsamples[0])
                orig_x = downsample_factor * x  # boundary with the maximal area (level 0)
                orig_y = downsample_factor * y

                if show:  # observe the result (in the current level)
                    cropped_img = np.array(slide.read_region(location=(orig_x, orig_y), level=curr_level, size=(w, h)))
                    gray_img = 1 - color.rgb2gray(color.rgba2rgb(cropped_img))

                    print(f"Image shape before cropping: {curr_level_img.shape[0]}, {curr_level_img.shape[1]}")
                    print(f"Image shape after cropping: {cropped_img.shape[0]}, {cropped_img.shape[1]}")

                    plt.figure()
                    plt.suptitle('Process 1: cropping and grayscale')
                    plt.subplot(1, 2, 1)
                    plt.imshow(curr_level_img)
                    plt.xlabel(f'level {curr_level} - before')
                    plt.subplot(1, 2, 2)
                    plt.imshow(gray_img, cmap="gray")
                    plt.xlabel(f'level {curr_level} - after')
                    plt.show()

                # process the given level images
                for level in all_levels:
                    print(f"Current level: {level}")

                    if level in levels:
                        rgba_output_path = os.path.join(paths.rgba_data_path, filename, f"level_{level}")
                        if not os.path.exists(rgba_output_path):
                            os.makedirs(rgba_output_path)
                    if level in register_levels:
                        grayscale_output_path = os.path.join(paths.grayscale_data_path, filename, f"level_{level}")
                        if not os.path.exists(grayscale_output_path):
                            os.makedirs(grayscale_output_path)

                    temp_downsample_factor = int(slide.level_downsamples[curr_level] / slide.level_downsamples[level])
                    temp_w = temp_downsample_factor * w
                    temp_h = temp_downsample_factor * h
                    factor = np.power(2, level)

                    if min(temp_w, temp_h) > max_patch_size:

                        # When min(height, width) > max_patch_size,
                        # split the image into several parts and save them respectively (prevent memory overflow)
                        n_row = int(np.ceil(temp_h / max_patch_size))
                        n_col = int(np.ceil(temp_w / max_patch_size))

                        step = 0
                        for row in range(n_row):
                            for col in range(n_col):
                                step += 1
                                if step % 10 == 0:
                                    print(f"Step: {step} / {n_row * n_col}")

                                x_pos = orig_x + factor * (col * max_patch_size)
                                y_pos = orig_y + factor * (row * max_patch_size)

                                if row == n_row - 1 and col == n_col - 1:
                                    curr_w = temp_w % max_patch_size if temp_w % max_patch_size != 0 else max_patch_size
                                    curr_h = temp_h % max_patch_size if temp_h % max_patch_size != 0 else max_patch_size
                                elif row == n_row - 1:
                                    curr_w = max_patch_size
                                    curr_h = temp_h % max_patch_size if temp_h % max_patch_size != 0 else max_patch_size
                                elif col == n_col - 1:
                                    curr_w = temp_w % max_patch_size if temp_w % max_patch_size != 0 else max_patch_size
                                    curr_h = max_patch_size
                                else:
                                    curr_w = max_patch_size
                                    curr_h = max_patch_size

                                # load and save the RGBA image patches
                                temp_img_patch = np.array(slide.read_region(location=(x_pos, y_pos), level=level, size=(curr_w, curr_h)))
                                if level in levels:
                                    if 'mha' in format:
                                        to_save_mha = sitk.GetImageFromArray(temp_img_patch[:, :, :3])
                                        to_save_mha_path = os.path.join(rgba_output_path, f"{row}_{col}.mha")
                                        sitk.WriteImage(to_save_mha, to_save_mha_path)
                                    if 'jpg' in format:
                                        to_save_jpg = temp_img_patch[:, :, :3].astype(np.uint8)
                                        to_save_jpg_path = os.path.join(rgba_output_path, f"{row}_{col}.jpg")
                                        mpimg.imsave(to_save_jpg_path, to_save_jpg)

                                # generate and save the grayscale image patches
                                if level in register_levels:
                                    temp_gray_patch = (1 - color.rgb2gray(color.rgba2rgb(temp_img_patch))).astype(np.float32)
                                    if 'mha' in format:
                                        to_save_mha = sitk.GetImageFromArray(temp_gray_patch)
                                        to_save_mha_path = os.path.join(grayscale_output_path, f"{row}_{col}.mha")
                                        sitk.WriteImage(to_save_mha, to_save_mha_path)
                                    if 'jpg' in format:
                                        to_save_jpg = sitk.GetImageFromArray((temp_gray_patch * 255).astype(np.ubyte))
                                        to_save_jpg_path = os.path.join(grayscale_output_path, f"{row}_{col}.jpg")
                                        sitk.WriteImage(to_save_jpg, to_save_jpg_path)

                    else:
                        temp_img = np.array(slide.read_region(location=(orig_x, orig_y), level=level, size=(temp_w, temp_h)))
                        if level in levels:
                            if 'mha' in format:
                                to_save_mha = sitk.GetImageFromArray(temp_img[:, :, :3])
                                to_save_mha_path = os.path.join(rgba_output_path, "0_0.mha")
                                sitk.WriteImage(to_save_mha, to_save_mha_path)
                            if 'jpg' in format:
                                to_save_jpg = temp_img[:, :, :3].astype(np.uint8)
                                to_save_jpg_path = os.path.join(rgba_output_path, "0_0.jpg")
                                mpimg.imsave(to_save_jpg_path, to_save_jpg)

                        if level in register_levels:
                            temp_gray_img = (1 - color.rgb2gray(color.rgba2rgb(temp_img))).astype(np.float32)
                            if 'mha' in format:
                                to_save_mha = sitk.GetImageFromArray(temp_gray_img)
                                to_save_mha_path = os.path.join(grayscale_output_path, "0_0.mha")
                                sitk.WriteImage(to_save_mha, to_save_mha_path)
                            if 'jpg' in format:
                                to_save_jpg = sitk.GetImageFromArray((temp_gray_img * 255).astype(np.ubyte))
                                to_save_jpg_path = os.path.join(grayscale_output_path, "0_0.jpg")
                                sitk.WriteImage(to_save_jpg, to_save_jpg_path)

                e_t = time.time()
                print(f"Time: {e_t - b_t}s")
                print()


def generate_image_pairs(image_pair_list):
    """
    Group the RGBA/Grayscale data into different image pairs,
    each image pair consists of a source and a target images.

    :param image_pair_list: list of image pairs
    """

    # folders of the original images
    rgba_data_path = paths.rgba_data_path
    grayscale_data_path = paths.grayscale_data_path

    # folders to save the image pairs
    rgba_pair_path = paths.rgba_pair_path
    grayscale_pair_path = paths.grayscale_pair_path
    if not os.path.exists(rgba_pair_path):
        os.makedirs(rgba_pair_path)
    if not os.path.exists(grayscale_pair_path):
        os.makedirs(grayscale_pair_path)

    for i in range(len(image_pair_list)):
        id = i + 1  # image pair id
        print(f"Generating the image pair {id}...")
        source_name = image_pair_list[i][0]
        target_name = image_pair_list[i][1]

        # paths of the original images
        rgba_source_path = os.path.join(rgba_data_path, source_name)
        rgba_target_path = os.path.join(rgba_data_path, target_name)
        grayscale_source_path = os.path.join(grayscale_data_path, source_name)
        grayscale_target_path = os.path.join(grayscale_data_path, target_name)

        # paths to save the image pairs
        rgba_source_output_path = os.path.join(rgba_pair_path, str(id), source_name + "_source")
        rgba_target_output_path = os.path.join(rgba_pair_path, str(id), target_name + "_target")
        grayscale_source_output_path = os.path.join(grayscale_pair_path, str(id), source_name + "_source")
        grayscale_target_output_path = os.path.join(grayscale_pair_path, str(id), target_name + "_target")

        shutil.copytree(rgba_source_path, rgba_source_output_path)
        shutil.copytree(rgba_target_path, rgba_target_output_path)
        shutil.copytree(grayscale_source_path, grayscale_source_output_path)
        shutil.copytree(grayscale_target_path, grayscale_target_output_path)




