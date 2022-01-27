import os
import random

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch

import utils
import paths


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CheckPoints:
    """
    Check point to save the model if a certain metric is improved.
    """
    def __init__(self, init_metric, path, model_name, mode="max", logger=None):
        self.mode = mode
        self.trial = 0
        self.path = path
        self.model_name = model_name
        if init_metric is not None:
            self.best_metric = init_metric
        else:
            self.best_metric = float("-inf") if mode == "max" else float("inf")
        self.save = False
        self.logger = logger

        print(f"Checkpoint: The initial metric is {self.best_metric}.")
        print()
        if logger is not None:
            logger.info(f"Checkpoint: The initial metric is {self.best_metric}.")
            logger.info(" ")

    def __call__(self, model, metric):
        if self.best_metric >= metric:
            if self.mode != "max":
                print(f"Checkpoint: best metric ({self.best_metric}) > current metric ({metric})")
                if self.logger is not None:
                    self.logger.info(f"Checkpoint: best metric ({self.best_metric}) > current metric ({metric})")

                if not os.path.isdir(self.path):
                    os.makedirs(self.path)

                self.best_metric = metric
                self.trial += 1
                torch.save(model.state_dict(), os.path.join(self.path, f"{self.model_name}_{self.trial}"))
                self.save = True

                print(f"Checkpoint: The params have been saved in <{self.model_name}_{self.trial}>.")
                if self.logger is not None:
                    self.logger.info(f"Checkpoint: The params have been saved in <{self.model_name}_{self.trial}>.")
            else:
                self.save = False

        elif self.best_metric < metric:
            if self.mode == "max":
                print()
                print(f"Checkpoint: best metric ({self.best_metric}) < current metric ({metric})")
                if self.logger is not None:
                    self.logger.info(" ")
                    self.logger.info(f"Checkpoint: best metric ({self.best_metric}) < current metric ({metric})")

                if not os.path.isdir(self.path):
                    os.makedirs(self.path)

                self.best_metric = metric
                self.trial += 1
                torch.save(model.state_dict(), os.path.join(self.path, f"{self.model_name}_{self.trial}"))
                self.save = True

                print(f"Checkpoint: The params have been saved in f'<{self.model_name}_{self.trial}>'.")
                if self.logger is not None:
                    self.logger.info(f"Checkpoint: The params have been saved in f'<{self.model_name}_{self.trial}>'.")
            else:
                self.save = False


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improved within certain epochs.
    """
    def __init__(self, patience=200, mode="max", logger=None):
        self.mode = mode
        self.patience = patience
        self.counter = 0
        self.best_metric = None
        self.switch = False
        self.logger = logger

    def __call__(self, metric):
        if self.best_metric is None:
            self.best_metric = metric
        elif self.best_metric >= metric:
            if self.mode == "max":
                self.counter += 1
            else:
                self.counter = 0
                self.best_metric = metric
        elif self.best_metric < metric:
            if self.mode == "max":
                self.counter = 0
                self.best_metric = metric
            else:
                self.counter += 1
        print(f"Early stopping: {self.counter} / {self.patience}")
        if self.logger is not None:
            self.logger.info(f"Early stopping: {self.counter} / {self.patience}")
        if self.counter >= self.patience:
            print("Early stopping: The count reaches the upper limit. The training is terminated.")
            if self.logger is not None:
                self.logger.info("Early stopping: The count reaches the upper limit. The training is terminated.")
            self.switch = True


def plot_losses(log_path, type):
    """
    Plot the train/val loss curves according to the log file and save

    :param log_path: str, path of the log file
    :param type: str, type of the model
    """
    figure_path = None

    if type == "affine_network":
        train_losses = []
        val_losses = []
        for line in open(log_path, 'r'):
            if "Train loss" in line:
                train_losses.append(float(line[12:]))
            if "Val loss" in line:
                val_losses.append(float(line[10:]))
            if "Figure" in line:
                figure_path = line[8:]

        length = min(len(train_losses), len(val_losses))
        train_losses, val_losses = train_losses[:length - 1], val_losses[:length - 1]

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(train_losses, "r-")
        plt.ylabel("Train Loss")
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.plot(val_losses, "b-")
        plt.ylabel("Val Loss")
        plt.grid(True)
        plt.xlabel("Epoch")

        if figure_path is None:
            raise UserWarning("No found the path to save figure.")
        else:
            if not os.path.isdir('/'.join(figure_path.split('/')[:-1])):
                os.makedirs('/'.join(figure_path.split('/')[:-1]))
            plt.savefig(figure_path[:-1], bbox_inches='tight', pad_inches=0)

    elif type == "nonrigid_network":
        train_losses_before = []
        train_losses_after = []
        val_losses_before = []
        val_losses_after = []
        train_regularization = []
        val_regularization = []
        for line in open(log_path, 'r'):
            if "Train loss before" in line:
                train_losses_before.append(float(line[19:]))
            if "Train loss after" in line:
                train_losses_after.append(float(line[18:]))
            if "Val loss before" in line:
                val_losses_before.append(float(line[17:]))
            if "Val loss after" in line:
                val_losses_after.append(float(line[16:]))
            if "Train regularization" in line:
                train_regularization.append(float(line[22:]))
            if "Val regularization" in line:
                train_regularization.append(float(line[20:]))
            if "Figure" in line:
                figure_path = line[8:]

        length = min(len(train_losses_before), len(val_losses_before))
        train_losses_before, val_losses_before = train_losses_before[:length - 1], val_losses_before[:length - 1]
        train_losses_after, val_losses_after = train_losses_after[:length - 1], val_losses_after[:length - 1]

        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(train_losses_before, "r-")
        plt.plot(train_losses_after, "b-")
        plt.ylabel("Train Loss")
        plt.legend(["before", "after"])
        plt.grid(True)
        plt.subplot(3, 1, 2)
        plt.plot(val_losses_before, "r-")
        plt.plot(val_losses_after, "b-")
        plt.ylabel("Val Loss")
        plt.legend(["before", "after"])
        plt.grid(True)
        plt.subplot(3, 1, 3)
        plt.plot(train_regularization, "r-")
        plt.plot(val_regularization, "b-")
        plt.ylabel("Regularization")
        plt.legend(["Train", "Val"])
        plt.xlabel("Epoch")

        if figure_path is None:
            raise UserWarning("No found the path to save figure.")
        else:
            if not os.path.isdir('/'.join(figure_path.split('/')[:-1])):
                os.makedirs('/'.join(figure_path.split('/')[:-1]))
            plt.savefig(figure_path[:-1], bbox_inches='tight', pad_inches=0)

    else:
        raise UserWarning("Please input an valid type of model.")


def create_training_data(input_path, train_path, val_path, level, mode="min", size=1024, show=False):
    """
    Create the training/validation data for the non-rigid registration network using our project dataset
    """
    id_list = sorted([int(id) for id in os.listdir(input_path)])
    train_ids = sorted(random.sample(id_list, k=int(np.ceil(0.8 * len(id_list)))))
    val_ids = [id for id in id_list if id not in train_ids]
    curr_train_id = 0
    curr_val_id = 0

    print(f"Training data id: {train_ids}")
    print(f"Validation data id: {val_ids}")
    print("Notice: Here are the ids of original data, later they will be changed.\n")

    for id in id_list:

        # load the current sample (image pair)
        current_id = str(id)
        sample_folder = os.path.join(input_path, current_id)
        source_name, target_name = None, None
        source_path, target_path = None, None
        for filename in os.listdir(sample_folder):
            if "source" in filename:
                source_name = filename
                source_path = os.path.join(sample_folder, filename)
            elif "target" in filename:
                target_name = filename
                target_path = os.path.join(sample_folder, filename)
            else:
                raise FileExistsError("Please make sure that there is a target image and a source image "
                                      "in each sample folder, with a note in the filename.")

        print(f"Current id: {id}")
        print(f"Source image path: {source_path}")
        print(f"Target image path: {target_path}")
        print(f"Level to use: level-{level}")

        source = utils.load_whole_image(filename=source_path, level=level, device=device)
        target = utils.load_whole_image(filename=target_path, level=level, device=device)

        # resample the image pair
        if mode == "min":
            new_shape = utils.calculate_new_shape((source.size(0), source.size(1)), size, mode="min")

            if min(new_shape) == min(source.size(0), source.size(1)):
                print("Resampling not required.")
                resampled_source = source
                resampled_target = target
            else:
                resample_factor = min(source.size(0), source.size(1)) / size
                gaussian_sigma = resample_factor / 1.05
                smoothed_source = utils.smooth_image_tensor(source, kernel_size=5, sigma=gaussian_sigma)
                smoothed_target = utils.smooth_image_tensor(target, kernel_size=5, sigma=gaussian_sigma)
                resampled_source = utils.resample_tensor(smoothed_source, new_size=new_shape, device=device)
                resampled_target = utils.resample_tensor(smoothed_target, new_size=new_shape, device=device)

        elif mode == "max":
            new_shape = utils.calculate_new_shape((source.size(0), source.size(1)), size, mode="max")

            if max(new_shape) == max(source.size(0), source.size(1)):
                print("Resampling not required.")
                resampled_source = source
                resampled_target = target
            else:
                resample_factor = max(source.size(0), source.size(1)) / size
                gaussian_sigma = resample_factor / 1.05
                smoothed_source = utils.smooth_image_tensor(source, kernel_size=5, sigma=gaussian_sigma)
                smoothed_target = utils.smooth_image_tensor(target, kernel_size=5, sigma=gaussian_sigma)
                resampled_source = utils.resample_tensor(smoothed_source, new_size=new_shape, device=device)
                resampled_target = utils.resample_tensor(smoothed_target, new_size=new_shape, device=device)

        elif mode == "original":
            resampled_source = source
            resampled_target = target

        print(f"Original shape: {source.size(0), source.size(1)}")
        print(f"Resampled shape: {resampled_source.size(0), resampled_source.size(1)}")

        resampled_source = resampled_source.cpu().numpy()
        resampled_target = resampled_target.cpu().numpy()

        if show:
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow(source.cpu().numpy(), cmap='gray')
            plt.subplot(2, 2, 2)
            plt.imshow(target.cpu().numpy(), cmap='gray')
            plt.subplot(2, 2, 3)
            plt.imshow(resampled_source, cmap='gray')
            plt.xlabel("source")
            plt.subplot(2, 2, 4)
            plt.imshow(resampled_target, cmap='gray')
            plt.xlabel("target")
            plt.show()

        # save the resampled image pair
        to_save_source_mha = sitk.GetImageFromArray(resampled_source)
        to_save_target_mha = sitk.GetImageFromArray(resampled_target)
        to_save_source_jpg = sitk.GetImageFromArray((resampled_source * 255).astype(np.ubyte))
        to_save_target_jpg = sitk.GetImageFromArray((resampled_target * 255).astype(np.ubyte))

        if id in train_ids:
            curr_train_id += 1
            output_path = os.path.join(train_path, str(curr_train_id))
        elif id in val_ids:
            curr_val_id += 1
            output_path = os.path.join(val_path, str(curr_val_id))

        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        print(f"Output_path: {output_path}")

        source_mha_output_path = os.path.join(output_path, f"{source_name}.mha")
        target_mha_output_path = os.path.join(output_path, f"{target_name}.mha")
        source_jpg_output_path = os.path.join(output_path, f"{source_name}.jpg")
        target_jpg_output_path = os.path.join(output_path, f"{target_name}.jpg")

        sitk.WriteImage(to_save_source_mha, source_mha_output_path)
        sitk.WriteImage(to_save_target_mha, target_mha_output_path)
        sitk.WriteImage(to_save_source_jpg, source_jpg_output_path)
        sitk.WriteImage(to_save_target_jpg, target_jpg_output_path)

        print()


if __name__ == '__main__':

    ### create the training/validation data for the registration network using our project dataset
    original_path = paths.la_output_folder
    training_path = paths.nonrigid_train_folder
    validation_path = paths.nonrigid_val_folder
    create_training_data(input_path=original_path, train_path=training_path, val_path=validation_path, level=4, mode="original", show=False)


