import numpy as np
import torch

import utils


def loss_global(source, target, device=torch.device("cpu"), **params):
    """
    Calculate different loss value of the input image pair:
    mse -- mean-squared error
    ncc -- normalized cross correlation
    nmi -- normalized mutual information
    """
    if len(source.size()) == 2:
        source = source.view(1, 1, source.size(0), source.size(1))  # (n_samples, 1, height, width)
    if len(target.size()) == 2:
        target = target.view(1, 1, target.size(0), target.size(1))  # (n_samples, 1, height, width)

    mode = params['mode']
    if mode == 'mse':
        return mse_losses_global(source, target, device=device, **params)
    elif mode == 'ncc':
        return ncc_losses_global(source, target, device=device, **params)
    elif mode == 'nmi':
        return nmi_losses_global(source, target, device=device, **params)
    else:
        raise KeyError("Unsupported distance measure.")


def mse_losses_global(sources, targets, device=torch.device("cpu"), **params):
    """
    Mean square error
    :param sources, targets: input image pairs, of size (num of samples, channels, width, height)
    :return: mse: mean squared error
    """
    size = sources.size(2) * sources.size(3)
    mse = (1 / size) * torch.sum((sources - targets) ** 2, dim=(1, 2, 3))
    mse = torch.mean(mse).to(device)
    return mse


def ncc_losses_global(sources, targets, device=torch.device("cpu"), **params):
    """
    Normalized cross-correlation
    Code from: https://github.com/MWod/DeepHistReg

    :param sources, targets: input image pairs, of size (num of samples, channels, width, height)
    :return: -ncc: negative normalized cross correlation
    """
    size = sources.size(2) * sources.size(3)
    sources_mean = torch.mean(sources, dim=(1, 2, 3)).view(sources.size(0), 1, 1, 1)
    targets_mean = torch.mean(targets, dim=(1, 2, 3)).view(sources.size(0), 1, 1, 1)
    sources_std = torch.std(sources, dim=(1, 2, 3)).view(sources.size(0), 1, 1, 1)
    targets_std = torch.std(targets, dim=(1, 2, 3)).view(sources.size(0), 1, 1, 1)
    ncc = (1 / size) * torch.sum(
        (sources - sources_mean) * (targets - targets_mean) / (sources_std * targets_std), dim=(1, 2, 3)
    )
    ncc = torch.mean(ncc)

    if ncc != ncc:
        return torch.autograd.Variable(torch.Tensor([1]), requires_grad=True).to(device)
    return -ncc


def nmi_losses_global(sources, targets, device=torch.device("cpu"), **params):
    """
    Normalized mutual information
    :param sources, targets: input image pairs, of size (num of samples, channels, width, height)
    :param params: dict of params,
             'epsilon': a constant to prevent numerical errors, default is 1.4e-45
    :return: -nmi: negative normalized mutual information
    """
    try:
        epsilon = params['epsilon']
    except UserWarning:
        epsilon = 1.4e-45
        print(f"The parameter <EPSILON> of NMI is set to  the default value {epsilon}.")

    nmi = list()
    for i in range(sources.size(0)):
        source = sources[i, :, :, :].contiguous().view(-1).cpu().numpy()
        target = targets[i, :, :, :].contiguous().view(-1).cpu().numpy()

        size = source.shape[0]
        p_source = np.histogram(source, 256, (0, 1))[0] / size
        p_target = np.histogram(target, 256, (0, 1))[0] / size
        h_source = - np.sum(p_source * np.log(p_source + epsilon))
        h_target = - np.sum(p_target * np.log(p_target + epsilon))

        h_total = np.histogram2d(source, target, 256, [[0, 1], [0, 1]])[0]
        h_total /= (1.0 * size)
        h_total = - np.sum(h_total * np.log(h_total + epsilon))
        nmi.append(2 * (h_source + h_target - h_total) / (h_source + h_target))

    nmi = torch.FloatTensor(nmi).to(device)
    nmi = torch.mean(nmi)
    return -nmi


def curvature_regularization(displacement_fields, device=torch.device("cpu")):
    """
    Calculate the curvature regularization term of the displacement field
    Code from: https://github.com/MWod/DeepHistReg

    :param displacement_fields: displacement fields in x- and y-direction, of size (num of samples, 2, width, height)
    """
    u_x = displacement_fields[:, 0, :, :].view(-1, 1, displacement_fields.size(2), displacement_fields.size(3))
    u_y = displacement_fields[:, 1, :, :].view(-1, 1, displacement_fields.size(2), displacement_fields.size(3))
    x_laplacian = utils.tensor_laplacian(u_x, device)[:, :, 1:-1, 1:-1]
    y_laplacian = utils.tensor_laplacian(u_y, device)[:, :, 1:-1, 1:-1]
    x_term = x_laplacian ** 2
    y_term = y_laplacian ** 2
    curvature = torch.mean(1 / 2 * (x_term + y_term))
    return curvature
