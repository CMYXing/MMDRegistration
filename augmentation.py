import random

import numpy as np
import matplotlib.pyplot as plt
import torch

import utils


def randrange(vmin, vmax):
    return vmin + random.random() * (vmax - vmin)


def random_affine_transform_matrix(shape, **params):
    """
    Randomly generate an affine transform matrix with the given size
    Code from: https://github.com/MWod/DeepHistReg
    """
    min_translation = params['min_translation']
    max_translation = params['max_translation']
    min_rotation = params['min_rotation']
    max_rotation = params['max_rotation']
    min_shear = params['min_shear']
    max_shear = params['max_shear']
    min_scale = params['min_scale']
    max_scale = params['max_scale']

    min_translation = min_translation * min(shape[0:2])
    max_translation = max_translation * max(shape[0:2])
    min_rotation = min_rotation * np.pi / 180
    max_rotation = max_rotation * np.pi / 180

    x_translation = randrange(min_translation, max_translation)
    y_translation = randrange(min_translation, max_translation)
    rotation = randrange(min_rotation, max_rotation)
    x_shear = randrange(min_shear, max_shear)
    y_shear = randrange(min_shear, max_shear)
    x_scale = randrange(min_scale, max_scale)
    y_scale = randrange(min_scale, max_scale)

    rigid_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation), x_translation],
        [np.sin(rotation), np.cos(rotation), y_translation],
        [0, 0, 1],
    ])
    cm1 = np.array([
        [1, 0, ((shape[0] - 1) / 2)],
        [0, 1, ((shape[1] - 1) / 2)],
        [0, 0, 1],
    ])
    cm2 = np.array([
        [1, 0, -((shape[0] - 1) / 2)],
        [0, 1, -((shape[1] - 1) / 2)],
        [0, 0, 1],
    ])
    rigid_matrix = cm1 @ rigid_matrix @ cm2

    shear_matrix = np.array([
        [1, x_shear, 0],
        [y_shear, 1, 0],
        [0, 0, 1],
    ])

    scale_matrix = np.array([
        [x_scale, 0, 0],
        [0, y_scale, 0],
        [0, 0, 1],
    ])

    all_matrices = [rigid_matrix, shear_matrix, scale_matrix]
    random.shuffle(all_matrices)
    transform = np.eye(3)
    for i in range(len(all_matrices)):
        transform = transform @ all_matrices[i]
    affine_transform = transform[0:2, :]
    return affine_transform


def affine_augmentation(params):
    def augmentation(source, target, device=torch.device("cpu"), show=False):
        """
        :param source: tensor, the source image
        :param target: tensor, the target image
        """
        affine_transform = random_affine_transform_matrix((source.size(0), source.size(1)), **params)
        theta_transform = torch.from_numpy(
            utils.numpy_affine2theta(affine_transform, (source.size(0), source.size(1))).astype(np.float32)).to(device)

        if random.random() > 0.5:
            transformed_source = utils.tensor_affine_transform(source, theta_transform)[0, 0, :, :]
            transformed_target = target
        else:
            transformed_source = source
            transformed_target = utils.tensor_affine_transform(target, theta_transform)[0, 0, :, :]

        if show:
            plt.figure()
            plt.suptitle("Randomly affine transform")
            plt.subplot(2, 2, 1)
            plt.imshow(source.cpu().numpy(), cmap='gray')
            plt.xlabel("source")
            plt.axis('off')
            plt.subplot(2, 2, 2)
            plt.imshow(target.cpu().numpy(), cmap='gray')
            plt.xlabel("target")
            plt.axis('off')
            plt.subplot(2, 2, 3)
            plt.imshow(transformed_source.cpu().numpy(), cmap='gray')
            plt.xlabel("transformed source")
            plt.axis('off')
            plt.subplot(2, 2, 4)
            plt.imshow(transformed_target.cpu().numpy(), cmap='gray')
            plt.xlabel("transformed target")
            plt.axis('off')
            plt.show()
            exit()

        return transformed_source, transformed_target
    return augmentation

