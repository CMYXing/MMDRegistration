import torch

import cost_functions as cf
import utils
import Parameter


def rigid_registration(source, target, angle_step, device=torch.device("cpu")):
    """
    Perform the rigid registration of the input image pair, which consists of translation and rotation transformations

    :param source: source image, tensor, of size (height, width)
    :param target: target image, tensor, of size (height, width)
    :param angle_step: stride of the angle for the rotation transformation, int
    :return: displacement_field: calculated displacement field, tensor, of size (2, height, width)
    """
    # similarity metric of the image pair
    cost_func = cf.loss_global
    cost_func_params = dict()
    cost_func_params["mode"] = Parameter.get_value("similarity_metric")

    # convert the input images to the binary form, which describes the tissue shape
    source_shape = utils.tensor_shape_detection(source, device=device)
    target_shape = utils.tensor_shape_detection(target, device=device)

    # calculate the initial similarity of image pair
    init_loss = cost_func(source_shape, target_shape, device=device, **cost_func_params)

    # initialize an identity transformation matrix (identity means that there is no change in the image)
    y_size, x_size = source_shape.size(0), source_shape.size(1)  # height and width of the image pair
    identity_transform = torch.tensor([
        [1, 0, 0.0],
        [0, 1, 0.0],
    ], dtype=torch.float32)
    identity_transform = utils.tensor_affine2theta(identity_transform, (y_size, x_size)).view(1, 2, 3).to(device)

    ### perform centroid translation
    # calculate the centroids of the source and target
    com_x_source, com_y_source = utils.center_of_mass(source_shape, device=device)  # centriod of the source
    com_x_target, com_y_target = utils.center_of_mass(target_shape, device=device)  # centriod of the target
    # generate the centroid translation matrix
    centroid_transform = torch.tensor([
        [1, 0, com_x_source - com_x_target],
        [0, 1, com_y_source - com_y_target],
    ], dtype=torch.float32)
    centroid_transform = utils.tensor_affine2theta(centroid_transform, (y_size, x_size)).view(1, 2, 3).to(device)
    # translate the source using the centroid transformation matrix
    translated_source_shape = utils.tensor_affine_transform(source_shape.view(1, 1, y_size, x_size), centroid_transform)[0, 0, :, :]
    # calculate the similarity of image pair after centroid translation
    # if the similarity increases, the image pair after translation is used in the rotation transformation; otherwise, the original image pair is used
    centroid_loss = cost_func(translated_source_shape, target_shape, device=device, **cost_func_params)
    centroid_translation = False if centroid_loss >= init_loss else True

    ### perform rotation transformation
    # initialize the rotation transformation params
    best_loss = centroid_loss if centroid_translation else init_loss  # the current optimal similarity of image pair
    found = False
    best_angle = 0
    # exhaustive search the best rotation angle:
    # (1) rotate the source image with each angle and calculate the similarity of rotated source and target
    # (2) select the angle with the highest similarity
    for i in range(1, 360, angle_step):
        if centroid_translation:
            # generate the rotation matrix around the center of mass
            rotation_transform = torch.tensor(utils.generate_rotation_matrix(i, com_x_target, com_y_target), dtype=torch.float32).to(device)
            transform = utils.compose_transforms(centroid_transform, utils.tensor_affine2theta(rotation_transform, (y_size, x_size))).view(1, 2, 3).to(device)
            # transform the source
            transformed_source_shape = utils.tensor_affine_transform(source_shape.view(1, 1, y_size, x_size), transform)[0, 0, :, :]
        else:
            # generate the rotation matrix around the center of the image
            rotation_transform = torch.tensor(utils.generate_rotation_matrix(i, x_size / 2, y_size / 2), dtype=torch.float32).to(device)
            transform = rotation_transform.view(1, 2, 3).to(device)
            # transform the source
            transformed_source_shape = utils.tensor_affine_transform(source_shape.view(1, 1, y_size, x_size), rotation_transform)[0, 0, :, :]
        # calculate the similarity of image pair after rotation
        current_loss = cost_func(transformed_source_shape, target_shape, device=device, **cost_func_params)
        if current_loss < best_loss:
            found = True
            best_loss = current_loss
            best_angle = i
            best_transform = transform

    ### generate the final transformation matrix and convert it to the displacement field of size (2, height, width)
    print("\n[INFO] Rigid registration")
    if found:
        if centroid_translation:
            # Case 1: initial similarity < centroid similarity and centroid similarity < rotation similarity
            # Transformation matrix: combination of the translation and rotation matrices
            print(f"Case 1: rotation loss ({best_loss}) < centroid loss ({centroid_loss})")
            print(f"        centroid loss ({centroid_loss}) < initial loss ({init_loss})")
            print(f"Rigid transformation: centroid translation + rotation ({best_angle}°)")
        else:
            # Case 2: initial similarity >= centroid similarity and initial similarity < rotation similarity
            # Transformation matrix: rotation matrix
            print(f"Case 2: rotation loss ({best_loss}) < initial loss ({init_loss})")
            print(f"        centroid loss ({centroid_loss}) >= initial loss ({init_loss})")
            print(f"Rigid transformation: rotation ({best_angle}°)")
        displacement_field = utils.transform_to_displacement_fields(source_shape.view(1, 1, y_size, x_size), best_transform, device=device)[0, :, :, :]
    else:
        if centroid_translation:
            # Case 3: initial similarity < centroid similarity and centroid similarity >= rotation similarity
            # Transformation matrix: translation matrix
            print(f"Case 3: rotation loss ({best_loss}) >= centroid loss ({centroid_loss})")
            print(f"        centroid loss ({centroid_loss}) < initial loss ({init_loss})")
            print("Rigid transformation: centroid translation")
            displacement_field = utils.transform_to_displacement_fields(source_shape.view(1, 1, y_size, x_size), centroid_transform, device=device)[0, :, :, :]
        else:
            # Case 4: initial similarity >= centroid similarity and initial similarity >= rotation similarity
            # Transformation matrix: identity matrix
            print(f"Case 4: rotation loss ({best_loss}) >= initial loss ({init_loss})")
            print(f"        centroid loss ({centroid_loss}) >= initial loss ({init_loss})")
            print("Rigid transformation: identity")
            displacement_field = utils.transform_to_displacement_fields(source_shape.view(1, 1, y_size, x_size), identity_transform, device=device)[0, :, :, :]

    # # resample the displacement field to the original size
    # displacement_field = utils.resample_displacement_fields(displacement_field, (1, 2, source.size(0), source.size(1)), padding_mode='border', device=device)[0, :, :, :]
    return displacement_field
