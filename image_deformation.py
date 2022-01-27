import os

import numpy as np
import matplotlib.image as mpimg
import torch
import torch.nn as nn

import utils
import Parameter


def generate_deformed_image(source_path, target_path, displacement_field, level, case_folder, device=torch.device("cpu")):
    """
    Generate and save the deformed image patches using the given displacement field

    :param source_path: str, path of the source image
    :param target_path: str, path of the target image
    :param displacement_field: tensor, the predicted displacement field
    :param level: tuple -- (source level, target level)
        source level: int, the level of input image pair
        target level: int, the level of image pair to generate and save
    """
    if len(displacement_field.size()) == 4:  # (1, 2, height, width)
        displacement_field = torch.squeeze(displacement_field, dim=0)  # (2, height, width)
    assert len(displacement_field.size()) == 3

    source_level = level[0]
    target_level = level[1]

    max_patch_size = Parameter.get_value("max_patch_size")
    input_path = Parameter.get_value("id_input_path")
    output_path = os.path.join(Parameter.get_value("id_output_path"))
    output_source_path = os.path.join(source_path.replace(input_path, output_path), case_folder)

    output_source_path = os.path.join(output_source_path, f"level_{target_level}")
    source_path = os.path.join(source_path, f"level_{target_level}")
    if not os.path.exists(output_source_path):
        os.makedirs(output_source_path)

    output_target_path = os.path.join(target_path.replace(input_path, output_path), case_folder)
    output_target_path = os.path.join(output_target_path, f"level_{target_level}")
    target_path = os.path.join(target_path, f"level_{target_level}")
    if not os.path.exists(output_target_path):
        os.makedirs(output_target_path)

    if source_level > target_level:
        size = int(max_patch_size / np.power(2, source_level - target_level))
        resample_factor = np.power(2, source_level - target_level)
    else:
        size = int(max_patch_size * np.power(2, target_level - source_level))
        resample_factor = 1 / np.power(2, target_level - source_level)

    # size of edge partitions
    rest_y = displacement_field.size(1) % size if displacement_field.size(1) % size != 0 else size
    rest_x = displacement_field.size(2) % size if displacement_field.size(2) % size != 0 else size

    # number of partitions
    n_row = int(np.ceil(displacement_field.size(1) / size))
    n_col = int(np.ceil(displacement_field.size(2) / size))

    if n_row == 1 or n_col == 1:
        n_row, n_col = 1, 1
        rest_y, rest_x = displacement_field.size(1), displacement_field.size(2)

    # pad the source level displacement field based the maximum value of displacement
    max_displacement = float(torch.abs(displacement_field).max())
    num_patch = int(np.ceil(max_displacement / size)) if n_row != 1 and n_col != 1 else 0
    pad_size = num_patch * size
    displacement_field = displacement_field.view(1, 2, displacement_field.size(1), displacement_field.size(2))
    pad = nn.ReplicationPad2d(padding=(pad_size, pad_size, pad_size, pad_size))
    displacement_field = pad(displacement_field)  # (1, 2, height, width)

    # partition resampling operation
    for row in range(n_row):
        for col in range(n_col):
            if (row == 1 and col == 3) or (row == 3 and col == 2):
                if n_row == 1 and n_col == 1:
                    temp_displacement_field = displacement_field
                else:
                    if row == n_row - 1 and col == n_col - 1:
                        temp_displacement_field = displacement_field[:, :,
                                                  row * size:,
                                                  col * size:]
                    elif row == n_row - 1 and (n_col - 1 - col) > num_patch:
                        temp_displacement_field = displacement_field[:, :,
                                                  row * size:,
                                                  col * size:(col + 1) * size + 2 * pad_size]
                    elif row == n_row - 1 and (n_col - 1 - col) <= num_patch:
                        temp_displacement_field = displacement_field[:, :,
                                                  row * size:,
                                                  col * size:(col + 1) * size + 2 * pad_size - size + rest_x]
                    elif col == n_col - 1 and (n_row - 1 - row) > num_patch:
                        temp_displacement_field = displacement_field[:, :,
                                                  row * size:(row + 1) * size + 2 * pad_size,
                                                  col * size:]
                    elif col == n_col - 1 and (n_row - 1 - row) <= num_patch:
                        temp_displacement_field = displacement_field[:, :,
                                                  row * size:(row + 1) * size + 2 * pad_size - size + rest_y,
                                                  col * size:]
                    elif (n_row - 1 - row) <= num_patch and (n_col - 1 - col) <= num_patch:
                        temp_displacement_field = displacement_field[:, :,
                                                  row * size:(row + 1) * size + 2 * pad_size - size + rest_y,
                                                  col * size:(col + 1) * size + 2 * pad_size - size + rest_x]
                    elif (n_row - 1 - row) <= num_patch:
                        temp_displacement_field = displacement_field[:, :,
                                                  row * size:(row + 1) * size + 2 * pad_size - size + rest_y,
                                                  col * size:(col + 1) * size + 2 * pad_size]
                    elif (n_col - 1 - col) <= num_patch:
                        temp_displacement_field = displacement_field[:, :,
                                                  row * size:(row + 1) * size + 2 * pad_size,
                                                  col * size:(col + 1) * size + 2 * pad_size - size + rest_x]
                    else:
                        temp_displacement_field = displacement_field[:, :,
                                                  row * size:(row + 1) * size + 2 * pad_size,
                                                  col * size:(col + 1) * size + 2 * pad_size]

                if n_row == 1 and n_col == 1:
                    curr_max_displacement = float(torch.abs(temp_displacement_field).max())
                else:
                    curr_max_displacement = float(torch.abs(temp_displacement_field[:, :, pad_size:-pad_size, pad_size:-pad_size]).max())
                curr_num_patch = int(np.ceil(curr_max_displacement / size)) if n_row != 1 and n_col != 1 else 0

                if curr_num_patch > 2:
                    device = torch.device("cpu")
                    temp_displacement_field = temp_displacement_field.cpu()
                else:
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                curr_pad_size = size * curr_num_patch
                spare = pad_size - curr_pad_size

                temp_source_patches = utils.load_image_patches(source_path, (row, col), curr_num_patch, max_patch_size, device=device)

                temp_y_size = int(temp_source_patches.size(0) / resample_factor)
                temp_x_size = int(temp_source_patches.size(1) / resample_factor)
                temp_displacement_field = temp_displacement_field[:, :, spare:spare+temp_y_size, spare:spare+temp_x_size]

                # resample the current displacement field partition
                if source_level == target_level:
                    resampled_displacement_field = temp_displacement_field
                else:
                    if n_row == 1 and n_col == 1:
                        resampled_displacement_field = utils.resample_displacement_fields(
                            temp_displacement_field, (1, 2, temp_source_patches.size(0), temp_source_patches.size(1)), padding_mode='border', device=device)
                    else:
                        resampled_displacement_field = utils.resample_displacement_fields(
                            temp_displacement_field, (1, 2, resample_factor * temp_displacement_field.size(2), resample_factor * temp_displacement_field.size(3)),
                            padding_mode='border', device=device)

                # warp image patches
                temp_source_patches = torch.unsqueeze(temp_source_patches.permute(2, 0, 1), dim=0)
                warped_source_patches = utils.warp_tensors(temp_source_patches, resampled_displacement_field, device=device)[0, :, :, :]

                # center crop the warped image patches
                y_b = int(resample_factor * curr_pad_size)
                x_b = int(resample_factor * curr_pad_size)
                if row == n_row - 1 and col == n_col - 1:
                    y_e = int(resample_factor * (curr_pad_size + rest_y))
                    x_e = int(resample_factor * (curr_pad_size + rest_x))
                elif row == n_row - 1:
                    y_e = int(resample_factor * (curr_pad_size + rest_y))
                    x_e = int(resample_factor * (curr_pad_size + size))
                elif col == n_col - 1:
                    y_e = int(resample_factor * (curr_pad_size + size))
                    x_e = int(resample_factor * (curr_pad_size + rest_x))
                else:
                    y_e = int(resample_factor * (curr_pad_size + size))
                    x_e = int(resample_factor * (curr_pad_size + size))
                warped_source_patch = warped_source_patches[:, y_b:y_e, x_b:x_e]

                # save the warped source patch
                warped_source_patch = warped_source_patch.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                mpimg.imsave(os.path.join(output_source_path, f"{row}_{col}.jpg"), warped_source_patch)

                # save the target patch
                target_patch = utils.load_image_patch(target_path, (row, col), device=device)
                target_patch = target_patch.cpu().numpy().astype(np.uint8)
                mpimg.imsave(os.path.join(output_target_path, f"{row}_{col}.jpg"), target_patch)
