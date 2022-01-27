import os
import time
import logging
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import data_loader as dl
import augmentation as aug
import cost_functions as cf
from network import global_affine_network as gan

import utils
import paths
from train_utils import EarlyStopping, CheckPoints
import Parameter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def training():
    """
    Train the affine registration network
    """
    curr_time = time.strftime("%m-%d-%H_%M_%S")

    ### Arguments
    parser = argparse.ArgumentParser()
    # training setting
    parser.add_argument('--epochs', dest="epochs", type=int, default=1000, help="Epoch of training")
    parser.add_argument('--patience', dest="patience", type=int, default=50, help="Patience of early stopping")
    parser.add_argument('--batch_size', dest="batch_size", type=int, default=1, help="Num of image pairs in batch")
    parser.add_argument('--initial_lr', dest="initial_lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument('--decay_rate', dest="decay_rate", type=float, default=0.995, help="Decay rate of learning rate with epoch")
    parser.add_argument('--initial_model', dest="initial_model", type=str, default=None, help="The pre-trained model name")
    # data setting
    parser.add_argument('--train_dir', dest="train_dir", type=str, default=paths.ga_train_folder, help="Directory for the training data")
    parser.add_argument('--val_dir', dest="val_dir", type=str, default=paths.ga_val_folder, help="Directory for the validation data")
    # save path setting
    parser.add_argument('--model_name', dest="model_name", type=str, default="global_affine_model", help="Model name")
    parser.add_argument('--model_dir', dest="model_dir", type=str, default=paths.bs_model_folder, help="Directory for saving the models")
    parser.add_argument('--log_dir', dest="log_dir", type=str, default=paths.logging_folder, help="Directory for saving the training logs")
    # data augmentation setting
    parser.add_argument('--augmentation', dest="augmentation", type=bool, default=True, help="Switch of data augmentation")
    parser.add_argument('--randomly_swap', dest="randomly_swap", type=bool, default=False, help="Switch of target and source swap")
    parser.add_argument('--min_translation', dest="min_translation", type=float, default=-0.005, help="The minimum allowed translation distance")
    parser.add_argument('--max_translation', dest="max_translation", type=float, default=0.005, help="The maximum allowed translation distance")
    parser.add_argument('--min_rotation', dest="min_rotation", type=float, default=-5, help="The minimum allowed rotation angle")
    parser.add_argument('--max_rotation', dest="max_rotation", type=float, default=5, help="The maximum allowed rotation angle")
    parser.add_argument('--min_shear', dest="min_shear", type=float, default=-1e-5, help="The minimum allowed shear rate")
    parser.add_argument('--max_shear', dest="max_shear", type=float, default=1e-5, help="The maximum allowed shear rate")
    parser.add_argument('--min_scale', dest="min_scale", type=float, default=0.995, help="The minimum allowed scale rate")
    parser.add_argument('--max_scale', dest="max_scale", type=float, default=1.005, help="The maximum allowed scale rate")
    args = parser.parse_args()

    ### Parse arguments
    # training setting
    epochs = args.epochs
    patience = args.patience
    batch_size = args.batch_size
    initial_lr = args.initial_lr
    decay_rate = args.decay_rate
    initial_model = args.initial_model
    if initial_model is not None:
        initial_model = os.path.join(args.model_dir, initial_model)
    # data setting
    train_folder = args.train_dir
    val_folder = args.val_dir
    # save path setting
    model_name = args.model_name
    model_save_folder = os.path.join(args.model_dir, curr_time)
    logging_folder = os.path.join(args.log_dir, "global_affine_network")
    if not os.path.isdir(logging_folder):
        os.makedirs(logging_folder)

    # data augmentation setting
    transforms = None
    augmentation = args.augmentation
    if augmentation:
        aug_params = dict()
        aug_params["min_translation"] = args.min_translation
        aug_params["max_translation"] = args.max_translation
        aug_params["min_rotation"] = args.min_rotation
        aug_params["max_rotation"] = args.max_rotation
        aug_params["min_shear"] = args.min_shear
        aug_params["max_shear"] = args.max_shear
        aug_params["min_scale"] = args.min_scale
        aug_params["max_scale"] = args.max_scale
        transforms = aug.affine_augmentation(aug_params)
    randomly_swap = args.randomly_swap

    ### Training setting
    model = gan.load_network(device, path=initial_model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: decay_rate ** epoch)
    cost_func = cf.loss_global
    cost_func_params = dict()
    cost_func_params["mode"] = "ncc"

    ### Create data loader
    training_loader = dl.MultiLevel_UnsupervisedLoader(train_folder, level=5, transforms=transforms, randomly_swap=randomly_swap, device=device)
    validation_loader = dl.MultiLevel_UnsupervisedLoader(val_folder, level=5, transforms=None, randomly_swap=False, device=device)
    training_dataloader = DataLoader(training_loader, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=dl.collate_to_list)
    validation_dataloader = DataLoader(validation_loader, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=dl.collate_to_list)
    training_size = len(training_dataloader.dataset)
    validation_size = len(validation_dataloader.dataset)

    ### Logging
    handler = logging.FileHandler(os.path.join(logging_folder, f"{curr_time}.txt"))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info(f"Model name: {model_name}")
    logger.info(" ")

    logger.info("<Related Files>")
    logger.info(f"Model: {model_save_folder}")
    logger.info(" ")

    logger.info("<Training Data>")
    logger.info(f"Training data: {train_folder}")
    logger.info(f"Validation data: {val_folder}")
    logger.info(f"Training size: {training_size}")
    logger.info(f"Validation size: {validation_size}")
    logger.info(" ")

    logger.info("<Training Params>")
    logger.info(f"Maximum iteration: {epochs}")
    logger.info(f"Early stopping patience: {patience}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Initial learning rate: {initial_lr}")
    logger.info(f"Decay rate of learning rate: {decay_rate}")
    logger.info(f"Initial model: {initial_model}")
    logger.info(f"Cost function: {cost_func_params['mode']}")
    logger.info(" ")

    logger.info("<Data Augmentation Params>")
    logger.info(f"Randomly swap (only for the train set): {randomly_swap}")
    logger.info(f"Data Augmentation: {augmentation}")
    if augmentation:
        logger.info(f"Minimum translation: {aug_params['min_translation']}")
        logger.info(f"Maximum translation: {aug_params['max_translation']}")
        logger.info(f"Minimum rotation: {aug_params['min_rotation']}")
        logger.info(f"Maximum rotation: {aug_params['max_rotation']}")
        logger.info(f"Minimum shear: {aug_params['min_shear']}")
        logger.info(f"Maximum shear: {aug_params['max_shear']}")
        logger.info(f"Minimum scale: {aug_params['min_scale']}")
        logger.info(f"Maximum scale: {aug_params['max_scale']}")
    logger.info(" ")

    ### Calculate initial loss
    initial_training_loss = 0.0
    initial_validation_loss = 0.0
    print("Calculate the initial loss...")
    for sources, targets in training_dataloader:
        with torch.set_grad_enabled(False):
            for i in range(len(sources)):
                source = sources[i]
                target = targets[i]
                source = source + 0.00001 * torch.randn((source.size(0), source.size(1))).to(device)
                target = target + 0.00001 * torch.randn((source.size(0), source.size(1))).to(device)
                loss = cost_func(source, target, device=device, **cost_func_params)
                initial_training_loss += loss.item()
    for sources, targets in validation_dataloader:
        with torch.set_grad_enabled(False):
            for i in range(len(sources)):
                source = sources[i]
                target = targets[i]
                source = source + 0.00001 * torch.randn((source.size(0), source.size(1))).to(device)
                target = target + 0.00001 * torch.randn((source.size(0), source.size(1))).to(device)
                loss = cost_func(source, target, device=device, **cost_func_params)
                initial_validation_loss += loss.item()
                print("Val loss per image pair: ", loss.item())

    print("Initial training loss: ", initial_training_loss / training_size)
    print("Initial validation loss: ", initial_validation_loss / validation_size)
    logger.info("-" * 40)
    logger.info(" ")
    logger.info(f"Initial training loss: {initial_training_loss / training_size}")
    logger.info(f"Initial validation loss: {initial_validation_loss / validation_size}")

    ### Start training
    check_points = CheckPoints(init_metric=initial_validation_loss / validation_size, path=model_save_folder, model_name=model_name, mode="min", logger=logger)
    early_stopping = EarlyStopping(patience=patience, mode="min", logger=logger)
    logger.info("Start training...")
    logger.info(" ")
    print("Start training...")
    print()

    for epoch in range(epochs):
        b_t = time.time()
        print(f"Current epoch:  {epoch + 1} / {epochs}")
        logger.info(f"Current epoch:  {epoch + 1} / {epochs}")

        # Training
        train_running_loss = 0.0
        model.train()
        for sources, targets in training_dataloader:
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                for i in range(len(sources)):
                    source = sources[i]
                    target = targets[i]
                    # add noise
                    source = source + 0.00001*torch.randn((source.size(0), source.size(1))).to(device)
                    target = target + 0.00001*torch.randn((source.size(0), source.size(1))).to(device)
                    # predict the transform matrix (forward)
                    pred_transform = model(source.view(1, 1, source.size(0), source.size(1)),
                                           target.view(1, 1, target.size(0), target.size(1)))
                    # calculate the loss
                    init_loss = cost_func(source, target, device=device, **cost_func_params)
                    transformed_source = utils.tensor_affine_transform(source, pred_transform)[0, 0, :, :]
                    curr_loss = cost_func(transformed_source, target, device=device, **cost_func_params)
                    loss = curr_loss - init_loss
                    train_running_loss += loss.item()
                    # backward
                    loss.backward()

                # update the parameters
                optimizer.step()

        avg_train_loss = train_running_loss / training_size
        print(f"Train loss: {avg_train_loss}")
        logger.info(f"Train loss: {avg_train_loss}")

        # Validation
        val_running_loss = 0.0
        model.eval()
        for sources, targets in validation_dataloader:
            with torch.set_grad_enabled(False):
                for i in range(len(sources)):
                    source = sources[i]
                    target = targets[i]
                    source = source + 0.00001*torch.randn((source.size(0), source.size(1))).to(device)
                    target = target + 0.00001*torch.randn((source.size(0), source.size(1))).to(device)
                    pred_transform = model(source.view(1, 1, source.size(0), source.size(1)),
                                           target.view(1, 1, target.size(0), target.size(1)))
                    transformed_source = utils.tensor_affine_transform(source, pred_transform)[0, 0, :, :]
                    loss = cost_func(transformed_source, target, device=device, **cost_func_params)
                    val_running_loss += loss.item()
                    print("Val loss per image pair: ", loss.item())

        scheduler.step()

        avg_val_loss = val_running_loss / validation_size
        print(f"Val loss: {avg_val_loss}")
        logger.info(f"Val loss: {avg_val_loss}")

        e_t = time.time()
        print(f"Epoch time: {e_t - b_t} seconds.")
        print(f"Estimated time to end: {(e_t - b_t) * (epochs - epoch)} seconds.")

        # Check point
        check_points(model=model, metric=avg_val_loss)

        # Early stopping
        early_stopping(avg_val_loss)
        if early_stopping.switch:
            break

        print()
        logger.info(" ")


def global_affine_registration(source, target, model, device=torch.device("cpu")):
    """
    Predict an global affine transform matrix using the given model

    :param source: source image, tensor, of size (height, width)
    :param target: target image, tensor, of size (height, width)
    :param model: global affine registration model
    :return: transform matrix: calculated transformation matrix, tensor, of size (1, 2, 3)
    """
    # similarity metric of the image pair
    cost_func = cf.loss_global
    cost_func_params = dict()
    cost_func_params["mode"] = Parameter.get_value("similarity_metric")

    # initialize an identity transform matrix
    identity_transform = torch.tensor([
        [1, 0, 0.0],
        [0, 1, 0.0],
    ], dtype=torch.float32)
    identity_transform = utils.tensor_affine2theta(identity_transform, (source.size(0), source.size(1))).view(1, 2, 3).to(device)

    with torch.set_grad_enabled(False):
        # predict the affine transform matrix using the pre-trained affine registration network
        calculated_transform = model(source.view(1, 1, source.size(0), source.size(1)), target.view(1, 1, target.size(0), target.size(1)))

        ### generate the final transformation matrix
        print("\n[INFO] Affine registration")
        # calculate the voxel intensity loss before and after affine registration
        transformed_source = utils.tensor_affine_transform(source.view(1, 1, source.size(0), source.size(1)), calculated_transform).view(source.size(0), source.size(1))
        voxel_loss_before = cost_func(source, target, device=device, **cost_func_params)
        voxel_loss_after = cost_func(transformed_source, target, device=device, **cost_func_params)
        # calculate the shape loss before and after affine registration
        source_shape = utils.tensor_shape_detection(source, device=device)
        target_shape = utils.tensor_shape_detection(target, device=device)
        transformed_source_shape = utils.tensor_shape_detection(transformed_source, device=device)
        shape_loss_before = cost_func(source_shape, target_shape, device=device, **cost_func_params)
        shape_loss_after = cost_func(transformed_source_shape, target_shape, device=device, **cost_func_params)
        # if the image similarity and shape similarity is higher than before and no significant decrease in shape similarity,
        # the predicted affine matrix is applied; otherwise, an identity matrix is returned
        if voxel_loss_after < voxel_loss_before:
            if shape_loss_after < 0.995 * shape_loss_before:
                print(f"Case 1: voxel loss after ({voxel_loss_after.item()}) < voxel loss before ({voxel_loss_before.item()})")
                print(f"        shape loss after ({shape_loss_after.item()}) < 0.995 * shape loss before {0.995 * shape_loss_before.item()})")
                print(f"Affine registration: affine")
                return calculated_transform
            else:
                print(f"Case 2: voxel loss after ({voxel_loss_after.item()}) < voxel loss before ({voxel_loss_before.item()})")
                print(f"        shape loss after ({shape_loss_after.item()}) >= 0.995 * shape loss before {0.995 * shape_loss_before.item()})")
                print(f"Affine registration: identity")
                return identity_transform
        else:
                print(f"Case 3: voxel loss after ({voxel_loss_after.item()}) >= voxel loss before ({voxel_loss_before.item()})")
                print(f"Affine registration: identity")
                return identity_transform


if __name__ == '__main__':

    training()