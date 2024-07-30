"""
author: Mihai Suteu
date: 15/05/19
"""


import os
import sys
import glob
from pathlib import Path
import h5py
import torch
import shutil
import random
import tarfile
import zipfile
import requests
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

# Refactored code based on https://github.com/xapharius/pytorch-nyuv2
class NYUv2Dataset(Dataset):
    """
    PyTorch wrapper for the NYUv2 dataset focused on multi-task learning.
    Data sources available: RGB, Semantic Segmentation, Surface Normals, Depth Images.
    If no transformation is provided, the image type will not be returned.

    ### Output
    All images are of size: 640 x 480

    1. RGB: 3 channel input image

    2. Semantic Segmentation: 1 channel representing one of the 14 (0 -
    background) classes. Conversion to int will happen automatically if
    transformation ends in a tensor.

    3. Surface Normals: 3 channels, with values in [0, 1].

    4. Depth Images: 1 channel with floats representing the distance in meters.
    Conversion will happen automatically if transformation ends in a tensor.
    """

    def __init__(self,
                 root_dir : str,
                 dataset_x : str,
                 dataset_y : str,
                 transform_x,
                 transform_y,
                 download : bool = False,
                 is_training : bool = True):

        # seg13, sn are not supported now.
        assert dataset_x == "rgb" or dataset_x == "depth" # or dataset_x == "seg13" or dataset_x == "sn" 
        assert dataset_y == "rgb" or dataset_y == "depth" # or dataset_y == "seg13" or dataset_y == "sn" 
        assert dataset_x != dataset_y

        super().__init__()
        self.root_dir : Path = Path(root_dir)
        self.x : list = []
        self.y : list = []
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.is_training : bool = is_training
        self.dataset_x : str = dataset_x
        self.dataset_y : str = dataset_y

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not complete." + " You can use download=True to download it"
            )

        self.load_data()

    def load_data(self):
        dir_name : str = "train" if self.is_training else "test"
        filepaths_x : list[str] = sorted(glob.glob((self.root_dir / f"{dir_name}_{self.dataset_x}").__str__() + "/*.png"))
        filepaths_y : list[str] = sorted(glob.glob((self.root_dir / f"{dir_name}_{self.dataset_y}").__str__() + "/*.png"))
        for filepath_x, filepath_y in zip(filepaths_x, filepaths_y):
            """
            https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html
            transforms.ToTensor scales RGB(or np.uint8) to [0, 1].
            BUT, the depth image we use will not scaled.
            """
            x = Image.open(filepath_x)
            y = Image.open(filepath_y)
            x = self.transform_x(x)
            y = self.transform_y(y)

            if self.dataset_x == "depth":
                x = x / 65000

            if self.dataset_y == "depth":
                y = y / 65000

            self.x.append(x)
            self.y.append(y)
    
    def __len__(self):
        assert len(self.x) == len(self.y)
        return len(self.x)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        return x, y


    def __repr__(self):
        fmt_str = f"Dataset {self.__class__.__name__}\n"
        fmt_str += f"    Number of data points: {self.__len__()}\n"
        fmt_str += f"    Split: {self._split}\n"
        fmt_str += f"    Root Location: {self.root_dir}\n"
        tmp = "    RGB Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.rgb_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Seg Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.seg_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    SN Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.sn_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Depth Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.depth_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str

    def _check_exists(self) -> bool:
        """
        Only checking for folder existence
        """
        try:
            for split in ["train", "test"]:
                # for part in ["rgb", "seg13", "sn", "depth"]:
                for part in ["rgb", "depth"]:
                    path = os.path.join(self.root_dir, f"{split}_{part}")
                    if not os.path.exists(path):
                        raise FileNotFoundError("Missing Folder")
        except FileNotFoundError as e:
            return False
        return True

    def download(self):
        if self._check_exists():
            return
        download_rgb(self.root_dir)
        # download_seg(self.root_dir)
        # download_sn(self.root_dir)
        download_depth(self.root_dir)
        print("Done!")


def download_rgb(root: str):
    train_url = "http://www.doc.ic.ac.uk/~ahanda/nyu_train_rgb.tgz"
    test_url = "http://www.doc.ic.ac.uk/~ahanda/nyu_test_rgb.tgz"

    def _proc(url: str, dst: str):
        if not os.path.exists(dst):
            tar = os.path.join(root, url.split("/")[-1])
            if not os.path.exists(tar):
                download_url(url, root)
            if os.path.exists(tar):
                _unpack(tar)
                _replace_folder(tar.rstrip(".tgz"), dst)
                _rename_files(dst, lambda x: x.split("_")[2])

    _proc(train_url, os.path.join(root, "train_rgb"))
    _proc(test_url, os.path.join(root, "test_rgb"))


def download_seg(root: str):
    train_url = "https://github.com/ankurhanda/nyuv2-meta-data/raw/master/train_labels_13/nyuv2_train_class13.tgz"
    test_url = "https://github.com/ankurhanda/nyuv2-meta-data/raw/master/test_labels_13/nyuv2_test_class13.tgz"

    def _proc(url: str, dst: str):
        if not os.path.exists(dst):
            tar = os.path.join(root, url.split("/")[-1])
            if not os.path.exists(tar):
                download_url(url, root)
            if os.path.exists(tar):
                _unpack(tar)
                _replace_folder(tar.rstrip(".tgz"), dst)
                _rename_files(dst, lambda x: x.split("_")[3])

    _proc(train_url, os.path.join(root, "train_seg13"))
    _proc(test_url, os.path.join(root, "test_seg13"))


def download_sn(root: str):
    url = "https://www.dropbox.com/s/dn5sxhlgml78l03/nyu_normals_gt.zip"
    train_dst = os.path.join(root, "train_sn")
    test_dst = os.path.join(root, "test_sn")

    if not os.path.exists(train_dst) or not os.path.exists(test_dst):
        tar = os.path.join(root, url.split("/")[-1])
        if not os.path.exists(tar):
            req = requests.get(url + "?dl=1") # dropbox
            with open(tar, 'wb') as f:
                f.write(req.content)
        if os.path.exists(tar):
            _unpack(tar)
            if not os.path.exists(train_dst):
                _replace_folder(
                    os.path.join(root, "nyu_normals_gt", "train"), train_dst
                )
                _rename_files(train_dst, lambda x: x[1:])
            if not os.path.exists(test_dst):
                _replace_folder(os.path.join(root, "nyu_normals_gt", "test"), test_dst)
                _rename_files(test_dst, lambda x: x[1:])
            shutil.rmtree(os.path.join(root, "nyu_normals_gt"))


def download_depth(root: str):
    url = (
        "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
    )
    train_dst = os.path.join(root, "train_depth")
    test_dst = os.path.join(root, "test_depth")

    if not os.path.exists(train_dst) or not os.path.exists(test_dst):
        tar = os.path.join(root, url.split("/")[-1])
        if not os.path.exists(tar):
            download_url(url, root)
        if os.path.exists(tar):
            train_ids = [
                f.split(".")[0] for f in os.listdir(os.path.join(root, "train_rgb"))
            ]
            _create_depth_files(tar, root, train_ids)


def _unpack(file: str):
    """
    Unpacks tar and zip, does nothing for any other type
    :param file: path of file
    """
    path = file.rsplit(".", 1)[0]

    if file.endswith(".tgz"):
        tar = tarfile.open(file, "r:gz")
        tar.extractall(path)
        tar.close()
    elif file.endswith(".zip"):
        zip = zipfile.ZipFile(file, "r")
        zip.extractall(path)
        zip.close()


def _rename_files(folder: str, rename_func: callable):
    """
    Renames all files inside a folder based on the passed rename function
    :param folder: path to folder that contains files
    :param rename_func: function renaming filename (not including path) str -> str
    """
    imgs_old = os.listdir(folder)
    imgs_new = [rename_func(file) for file in imgs_old]
    for img_old, img_new in zip(imgs_old, imgs_new):
        shutil.move(os.path.join(folder, img_old), os.path.join(folder, img_new))


def _replace_folder(src: str, dst: str):
    """
    Rename src into dst, replacing/overwriting dst if it exists.
    """
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.move(src, dst)


def _create_depth_files(mat_file: str, root: str, train_ids: list):
    """
    Extract the depth arrays from the mat file into images
    :param mat_file: path to the official labelled dataset .mat file
    :param root: The root directory of the dataset
    :param train_ids: the IDs of the training images as string (for splitting)
    """
    os.mkdir(os.path.join(root, "train_depth"))
    os.mkdir(os.path.join(root, "test_depth"))
    train_ids = set(train_ids)

    depths = h5py.File(mat_file, "r")["depths"]
    for i in range(len(depths)):
        img = (depths[i] * 6500).astype(np.uint16).T # max = 10. max(uint16) = 65535
        id_ = str(i + 1).zfill(4)
        folder = "train" if id_ in train_ids else "test"
        save_path = os.path.join(root, f"{folder}_depth", id_ + ".png")
        Image.fromarray(img).save(save_path)