import os
import random

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

import utils


def collate_to_list(batch):
    sources = [item[0].view(item[0].size(0), item[0].size(1)) for item in batch]
    targets = [item[1].view(item[1].size(0), item[1].size(1)) for item in batch]
    return sources, targets


class UnsupervisedLoader(Dataset):
    """
    This data loader class is used for the training dataset.
    """
    def __init__(self, data_path, transforms=None, randomly_swap=False, device=torch.device("cpu")):
        self.data_path = data_path
        self.id_list = [str(id) for id in sorted([int(id) for id in os.listdir(self.data_path)])]
        self.transforms = transforms
        self.randomly_swap = randomly_swap
        self.device = device

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):

        current_id = self.id_list[idx]
        sample_folder = os.path.join(self.data_path, current_id)

        source_path, target_path = None, None
        for filename in os.listdir(sample_folder):
            if "source.mha" in filename:
                source_path = os.path.join(sample_folder, filename)
            elif "target.mha" in filename:
                target_path = os.path.join(sample_folder, filename)

        assert source_path is not None
        assert target_path is not None
        source = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(source_path)).astype(np.float32)).to(self.device)
        target = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(target_path)).astype(np.float32)).to(self.device)

        if self.transforms is not None:
            source, target = self.transforms(source, target, device=self.device, show=False)

        if self.randomly_swap:
            if random.random() <= 0.5:
                source, target = target, source

        return source, target


class MultiLevel_UnsupervisedLoader(Dataset):
    """
    This data loader is used for our project dataset, where each image has multiple levels.
    """
    def __init__(self, data_path, level=4, transforms=None, randomly_swap=False, device=torch.device("cpu")):
        self.data_path = data_path
        self.level = level
        self.id_list = [str(id) for id in sorted([int(id) for id in os.listdir(self.data_path)])]
        self.transforms = transforms
        self.randomly_swap = randomly_swap
        self.device = device

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):

        current_id = self.id_list[idx]
        sample_folder = os.path.join(self.data_path, current_id)

        source_path, target_path = None, None
        for filename in os.listdir(sample_folder):
            if "source" in filename:
                source_path = os.path.join(sample_folder, filename)
            elif "target" in filename:
                target_path = os.path.join(sample_folder, filename)

        assert source_path is not None
        assert target_path is not None

        source = utils.load_whole_image(filename=source_path, level=self.level, device=self.device)
        target = utils.load_whole_image(filename=target_path, level=self.level, device=self.device)

        if self.transforms is not None:
            source, target = self.transforms(source, target, device=self.device, show=False)

        if self.randomly_swap:
            if random.random() <= 0.5:
                source, target = target, source

        return source, target