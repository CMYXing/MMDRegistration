import os
import time
import logging
import argparse

import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import data_loader as dl
import cost_functions as cf
from train_utils import EarlyStopping, CheckPoints
from network import mm_nonrigid_network as mmnr

import utils
import paths
import Parameter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def training():
    """
    Train the multi-magnification non-rigid registration network
    """
    curr_time = time.strftime("%m-%d-%H_%M_%S")

    ### Arguments
    parser = argparse.ArgumentParser()

    # training setting
    parser.add_argument('--epochs', dest="epochs", type=int, default=1000, help="Epoch of training")
    parser.add_argument('--patience', dest="patience", type=int, default=50, help="Patience of early stopping")
    parser.add_argument('--batch_size', dest="batch_size", type=int, default=1, help="Num of image pairs in batch")
    parser.add_argument('--num_of_patches', dest="num_of_patches", type=int, default=4, help="Num of patches in each group")
    parser.add_argument('--initial_lr', dest="initial_lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument('--decay_rate', dest="decay_rate", type=float, default=0.95, help="Decay rate of learning rate with epoch")
    parser.add_argument('--stride', dest="stride", type=int, default=112, help="Stride of extraction of patch sets")
    parser.add_argument('--patch_size', dest="patch_size", type=int, default=224, help="Input patch size of network")
    parser.add_argument('--magnifications_num', dest="magnifications_num", type=int, default=2, help="Num of patches with different magnifications in each patch set")
    parser.add_argument('--inner_iterations', dest="inner_iterations", type=int, default=2, help="Num of iterations per registration")
    parser.add_argument('--alpha', dest="alpha", type=int, default=60, help="Coefficient of the curvature regularization term")
    parser.add_argument('--initial_model', dest="initial_model", type=str, default=None, help="The pre-trained model name")
    parser.add_argument('--cycle_consistent', dest="cycle_consistent", type=bool, default=False, help="Switch of cycle consistent training")

    # data setting
    parser.add_argument('--train_dir', dest="train_dir", type=str, default=paths.nr_train_folder, help="Directory for the training data")
    parser.add_argument('--val_dir', dest="val_dir", type=str, default=paths.nr_val_folder, help="Directory for the validation data")

    # save path setting
    parser.add_argument('--model_name', dest="model_name", type=str, default="mm_nonrigid_model", help="Model name")
    parser.add_argument('--model_dir', dest="model_dir", type=str, default=paths.nr_model_folder, help="Directory for saving the models")
    parser.add_argument('--log_dir', dest="log_dir", type=str, default=paths.logging_folder, help="Directory for saving the training logs")

    args = parser.parse_args()

    ### Parse arguments
    epochs = args.epochs
    patience = args.patience
    batch_size = args.batch_size
    num_of_patches = args.num_of_patches
    initial_lr = args.initial_lr
    decay_rate = args.decay_rate
    stride = args.stride
    patch_size = args.patch_size
    magnifications_num = args.magnifications_num
    inner_iterations = args.inner_iterations
    alpha = args.alpha
    cycle_consistent = args.cycle_consistent

    initial_model = args.initial_model
    if initial_model is not None:
        initial_model = os.path.join(args.model_dir, initial_model)

    train_folder = args.train_dir
    val_folder = args.val_dir

    model_name = args.model_name
    model_save_folder = os.path.join(args.model_dir, curr_time)
    logging_folder = os.path.join(args.log_dir, "nonrigid_network")
    if not os.path.isdir(logging_folder):
        os.makedirs(logging_folder)

    ### Training setting
    model = mmnr.load_network(device, path=initial_model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: decay_rate ** epoch)

    cost_func_params = dict()
    cost_func_params["mode"] = "ncc"
    cost_func_params["regularization"] = "curvature"
    cost_func_params["alpha"] = alpha
    cost_func = cf.loss_global
    regularization_func = cf.curvature_regularization

    ### Create data loader
    transforms = None
    training_loader = dl.MultiLevel_UnsupervisedLoader(train_folder, level=4, transforms=transforms, randomly_swap=False, device=device)
    validation_loader = dl.MultiLevel_UnsupervisedLoader(val_folder, level=4, transforms=None, randomly_swap=False, device=device)
    training_dataloader = DataLoader(training_loader, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=dl.collate_to_list)
    validation_dataloader = DataLoader(validation_loader, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=dl.collate_to_list)
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
    logger.info(f"Num of patches in each group: {num_of_patches}")
    logger.info(f"Initial learning rate: {initial_lr}")
    logger.info(f"Decay rate of learning rate: {decay_rate}")
    logger.info(f"Stride: {stride}")
    logger.info(f"Patch size: {patch_size}")
    logger.info(f"Num of patches in each set: {magnifications_num}")
    logger.info(f"Num of iterations per registration: {inner_iterations}")
    logger.info(f"Initial model: {initial_model}")
    logger.info(f"Cycle consistent training: {cycle_consistent}")
    logger.info(" ")

    logger.info("<Cost function>")
    logger.info(f"Similarity metric: {cost_func_params['mode']}")
    logger.info(f"Regularization term: {cost_func_params['regularization']}")
    logger.info(f"Alpha: {cost_func_params['alpha']}")
    logger.info(" ")

    ### Start training
    logger.info("Start training...")
    logger.info(" ")
    print("Start training...")
    print()

    check_points = CheckPoints(init_metric=None, path=model_save_folder, model_name=model_name, mode="min", logger=logger)
    early_stopping = EarlyStopping(patience=patience, mode="min", logger=logger)

    print_step = 1
    cropping_rate = int(patch_size / stride)

    for epoch in range(epochs):
        b_t = time.time()
        print(f"Current epoch:  {epoch + 1} / {epochs}")
        logger.info(f"Current epoch:  {epoch + 1} / {epochs}")

        # training
        training_cost_before = 0.0
        training_cost_after = 0.0
        training_regularization = 0.0
        model.train()
        current_image = 0
        for sources, targets in training_dataloader:
            if current_image % print_step == 0:
                print("Training images: ", current_image + 1, "/", training_size)
            current_image += len(sources)
            for i in range(len(sources)):
                source = sources[i]  # (height, width)
                target = targets[i]

                for inner_iter in range(inner_iterations):
                    with torch.set_grad_enabled(False):
                        if inner_iter == 0:
                            # initialize the inner displacement field
                            inner_displacement_field = torch.zeros(1, 2, target.size(0), target.size(1)).to(device)
                            # unfold the source and target image (exist overlapping)
                            unfolded_source, padding_tuple = utils.unfold(source, stride, patch_size, magnifications_num, device=device)
                            unfolded_target, _ = utils.unfold(target, stride, patch_size, magnifications_num, device=device)
                        else:
                            # warp the source image based on the current predicted displacement field
                            warped_source = utils.warp_tensors(source, inner_displacement_field, device=device)
                            # unfold the warped source image
                            unfolded_source, padding_tuple = utils.unfold(warped_source, stride, patch_size, magnifications_num, device=device)

                        num_patch_sets = unfolded_source.size(0)
                        iters = math.ceil(num_patch_sets / num_of_patches)
                        all_displacement_fields = torch.Tensor([]).to(device)

                    for j in range(iters):
                        with torch.set_grad_enabled(False):
                            if j == iters - 1:
                                sm_sps = unfolded_source[j * num_of_patches:, :, :, :]
                                sm_tps = unfolded_target[j * num_of_patches:, :, :, :]
                            else:
                                sm_sps = unfolded_source[j * num_of_patches:(j + 1) * num_of_patches, :, :, :]
                                sm_tps = unfolded_target[j * num_of_patches:(j + 1) * num_of_patches, :, :, :]

                            # generate the set of multi-magnification patches
                            mm_sps = utils.create_multi_magnification_patches(sm_sps, patch_size, magnifications_num, device=device)
                            mm_tps = utils.create_multi_magnification_patches(sm_tps, patch_size, magnifications_num, device=device)

                            # skip the background image patches
                            if 0 in torch.std(mm_sps[:, 0, :, :], dim=(1, 2)) or 0 in torch.std(mm_tps[:, 0, :, :], dim=(1, 2)):
                                all_displacement_fields = torch.cat((all_displacement_fields, torch.zeros((mm_sps.size(0), 2, int(mm_sps.size(2) / cropping_rate), int(mm_sps.size(3) / cropping_rate))).to(device)))
                                continue

                            # add noise
                            mm_sps = mm_sps + torch.rand(mm_sps.size()).to(device) * 0.0000001
                            mm_tps = mm_tps + torch.rand(mm_tps.size()).to(device) * 0.0000001

                        optimizer.zero_grad()
                        with torch.set_grad_enabled(True):
                            # cycle consistent training (forward)
                            displacement_fields = model(mm_sps, mm_tps)
                            mm_tsps = utils.warp_tensors(mm_sps, displacement_fields, device=device)
                            if cycle_consistent:
                                inv_displacement_fields = model(mm_tsps, mm_sps)
                                mm_rsps = utils.warp_tensors(mm_tsps, inv_displacement_fields, device=device)

                            # concatenate the predicted displacement field patches
                            if stride != patch_size:
                                cropped_displacement_fields = utils.center_crop_tensor(displacement_fields.clone(), cropping_rate=cropping_rate, device=device)
                                all_displacement_fields = torch.cat((all_displacement_fields, cropped_displacement_fields))
                            else:
                                all_displacement_fields = torch.cat((all_displacement_fields, displacement_fields))

                            # calculate loss
                            mm_tps = torch.unsqueeze(mm_tps[:, 0, :, :], dim=1).to(device)
                            mm_tsps = torch.unsqueeze(mm_tsps[:, 0, :, :], dim=1).to(device)
                            cost = cost_func(mm_tsps, mm_tps, device, **cost_func_params)
                            reg = alpha * regularization_func(displacement_fields, device=device)
                            loss = cost + reg
                            if cycle_consistent:
                                mm_sps = torch.unsqueeze(mm_sps[:, 0, :, :], dim=1).to(device)
                                mm_rsps = torch.unsqueeze(mm_rsps[:, 0, :, :], dim=1).to(device)
                                inv_cost = cost_func(mm_rsps, mm_sps, device, **cost_func_params)
                                inv_reg = alpha * regularization_func(inv_displacement_fields, device=device)
                                loss += inv_cost + inv_reg

                            # backward propagate
                            loss.backward()
                            optimizer.step()

                    with torch.set_grad_enabled(False):
                        # restore the displacement field to a size correspondding to the original image pair
                        all_displacement_fields = utils.restore_displacement_field(all_displacement_fields, (target.size(0), target.size(1)), padding_tuple, stride, device=device)
                        smoothed_displacement_field = utils.smooth_displacement_fields(all_displacement_fields, kernel_size=75, gaussian=False, device=device)
                        # merge the current displacement field with the displacement field in the previous iteration
                        inner_displacement_field = utils.compose_displacement_fields(inner_displacement_field, smoothed_displacement_field, device=device)

                # calculate the loss of image pair before and after the non-rigid registration
                cost_before = cost_func(source, target, device=device, **cost_func_params)
                cost_after = cost_func(utils.warp_tensors(source, inner_displacement_field, device=device), target, device=device, **cost_func_params)
                cost_reg = regularization_func(inner_displacement_field, device=device)
                training_cost_before += cost_before.item()
                training_cost_after += cost_after.item()
                training_regularization += cost_reg.item()
                print(f"Train loss before: {cost_before.item()}")
                print(f"Train loss after: {cost_after.item()}")

        print(f"Train loss before: {training_cost_before / training_size}")
        print(f"Train loss after: {training_cost_after / training_size}")
        print(f"Train regularization: {training_regularization / training_size}")
        logger.info(f"Train loss before: {training_cost_before / training_size}")
        logger.info(f"Train loss after: {training_cost_after / training_size}")
        logger.info(f"Train regularization: {training_regularization / training_size}")

        # validation
        validation_cost_before = 0.0
        validation_cost_after = 0.0
        validation_regularization = 0.0
        model.eval()
        current_image = 0
        for sources, targets in validation_dataloader:
            if current_image % print_step == 0:
                print("Validation images: ", current_image + 1, "/", validation_size)
            current_image += len(sources)
            for i in range(len(sources)):
                source = sources[i]
                target = targets[i]

                for inner_iter in range(inner_iterations):
                    with torch.set_grad_enabled(False):
                        if inner_iter == 0:
                            # initialize the inner displacement field
                            inner_displacement_field = torch.zeros(1, 2, target.size(0), target.size(1)).to(device)
                            # unfold the source and target image (exist overlapping)
                            unfolded_source, padding_tuple = utils.unfold(source, stride, patch_size, magnifications_num, device=device)
                            unfolded_target, _ = utils.unfold(target, stride, patch_size, magnifications_num, device=device)
                        else:
                            # warp the source image based on the current predicted displacement field
                            warped_source = utils.warp_tensors(source, inner_displacement_field, device=device)
                            # unfold the warped source image
                            unfolded_source, padding_tuple = utils.unfold(warped_source, stride, patch_size, magnifications_num, device=device)

                        num_patch_sets = unfolded_source.size(0)
                        iters = math.ceil(num_patch_sets / num_of_patches)
                        all_displacement_fields = torch.Tensor([]).to(device)

                    for j in range(iters):
                        with torch.set_grad_enabled(False):
                            if j == iters - 1:
                                sm_sps = unfolded_source[j * num_of_patches:, :, :, :]
                                sm_tps = unfolded_target[j * num_of_patches:, :, :, :]
                            else:
                                sm_sps = unfolded_source[j * num_of_patches:(j + 1) * num_of_patches, :, :, :]
                                sm_tps = unfolded_target[j * num_of_patches:(j + 1) * num_of_patches, :, :, :]

                            # generate the set of multi-magnification patches
                            mm_sps = utils.create_multi_magnification_patches(sm_sps, patch_size, magnifications_num, device=device)
                            mm_tps = utils.create_multi_magnification_patches(sm_tps, patch_size, magnifications_num, device=device)

                            mm_sps = mm_sps + torch.rand(mm_sps.size()).to(device) * 0.0000001
                            mm_tps = mm_tps + torch.rand(mm_tps.size()).to(device) * 0.0000001

                            displacement_fields = model(mm_sps, mm_tps)

                            if stride != patch_size:
                                cropped_displacement_fields = utils.center_crop_tensor(displacement_fields.clone(), cropping_rate=cropping_rate, device=device)
                                all_displacement_fields = torch.cat((all_displacement_fields, cropped_displacement_fields))
                            else:
                                all_displacement_fields = torch.cat((all_displacement_fields, displacement_fields))

                    with torch.set_grad_enabled(False):
                        all_displacement_fields = utils.restore_displacement_field(all_displacement_fields, (target.size(0), target.size(1)), padding_tuple, stride, device=device)
                        smoothed_displacement_field = utils.smooth_displacement_fields(all_displacement_fields, kernel_size=7, gaussian=False, device=device)
                        inner_displacement_field = utils.compose_displacement_fields(inner_displacement_field, smoothed_displacement_field, device=device)

                cost_before = cost_func(source, target, device=device, **cost_func_params)
                cost_after = cost_func(utils.warp_tensors(source, inner_displacement_field, device=device), target, device=device, **cost_func_params)
                cost_reg = regularization_func(inner_displacement_field, device=device)
                validation_cost_before += cost_before.item()
                validation_cost_after += cost_after.item()
                validation_regularization += cost_reg.item()
                print(f"Val loss before: {cost_before.item()}")
                print(f"Val loss after: {cost_after.item()}")

        print(f"Val loss before: {validation_cost_before / validation_size}")
        print(f"Val loss after: {validation_cost_after / validation_size}")
        print(f"Val regularization: {validation_regularization / validation_size}")
        logger.info(f"Val loss before: {validation_cost_before / validation_size}")
        logger.info(f"Val loss after: {validation_cost_after / validation_size}")
        logger.info(f"Val regularization: {validation_regularization / validation_size}")

        scheduler.step()

        e_t = time.time()
        print(f"Epoch time: {e_t - b_t} seconds.")
        print(f"Estimated time to end: {(e_t - b_t) * (epochs - epoch)} seconds.")

        # Check point
        check_points(model=model, metric=validation_cost_after / validation_size)

        # Early stopping
        early_stopping(metric=validation_cost_after / validation_size)
        if early_stopping.switch:
            break

        print()
        logger.info(" ")


def nonrigid_registration(source, target, model, inner_iterations, device=torch.device("cpu")):
    
    print("\n[INFO] Non-rigid registration")

    stride = Parameter.get_value("nr_stride")
    patch_size = Parameter.get_value("nr_patch_size")
    magnifications_num = Parameter.get_value("nr_magnifications_num")
    num_of_patches = 4

    for inner_iter in range(inner_iterations):
        with torch.set_grad_enabled(False):
            if inner_iter == 0:
                # initialize the inner displacement field
                inner_displacement_field = torch.zeros(1, 2, target.size(0), target.size(1)).to(device)
                # unfold the source and target image (exist overlapping)
                unfolded_source, padding_tuple = utils.unfold(source, stride, patch_size, magnifications_num, device=device)
                unfolded_target, _ = utils.unfold(target, stride, patch_size, magnifications_num, device=device)
            else:
                # warp the source image based on the current predicted displacement field
                warped_source = utils.warp_tensors(source, inner_displacement_field, device=device)
                # unfold the warped source image
                unfolded_source, padding_tuple = utils.unfold(warped_source, stride, patch_size, magnifications_num, device=device)

            num_patch_sets = unfolded_source.size(0)
            iters = math.ceil(num_patch_sets / num_of_patches)
            all_displacement_fields = torch.Tensor([]).to(device)

        for j in range(iters):
            with torch.set_grad_enabled(False):
                if j == iters - 1:
                    sm_sps = unfolded_source[j * num_of_patches:, :, :, :]
                    sm_tps = unfolded_target[j * num_of_patches:, :, :, :]
                else:
                    sm_sps = unfolded_source[j * num_of_patches:(j + 1) * num_of_patches, :, :, :]
                    sm_tps = unfolded_target[j * num_of_patches:(j + 1) * num_of_patches, :, :, :]

                # generate the set of multi-magnification patches
                mm_sps = utils.create_multi_magnification_patches(sm_sps, patch_size, magnifications_num, device=device)
                mm_tps = utils.create_multi_magnification_patches(sm_tps, patch_size, magnifications_num, device=device)

                displacement_fields = model(mm_sps, mm_tps)

                if stride != patch_size:
                    cropped_displacement_fields = utils.center_crop_tensor(displacement_fields.clone(), cropping_rate=int(patch_size / stride), device=device)
                    all_displacement_fields = torch.cat((all_displacement_fields, cropped_displacement_fields))
                else:
                    all_displacement_fields = torch.cat((all_displacement_fields, displacement_fields))

        with torch.set_grad_enabled(False):
            all_displacement_fields = utils.restore_displacement_field(all_displacement_fields, (target.size(0), target.size(1)), padding_tuple, stride, device=device)
            smoothed_displacement_field = utils.smooth_displacement_fields(all_displacement_fields, kernel_size=7, gaussian=False, device=device)
            inner_displacement_field = utils.compose_displacement_fields(inner_displacement_field, smoothed_displacement_field, device=device)

    displacement_field = utils.resample_displacement_fields(inner_displacement_field, (1, 2, source.size(0), source.size(1)), device=device)[0, :, :, :]

    return displacement_field



if __name__ == '__main__':
    training()
