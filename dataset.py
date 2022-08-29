import os
import random

import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset

from encodings.runlength import rle_decode


class ImgDataset(Dataset):
    def __init__(self,
                 img_dpath,
                 img_fnames,
                 img_transform,
                 mask_encodings,
                 mask_size,
                 mask_transform=None):
        """
        Creates an instance of the dataset of images
        :param img_dpath: path to the folder with images
        :param img_fnames: the list of names of images to process
        :param img_transform: transformations for the images
        :param mask_encodings: pd.DataFrame with run-length encodings of image masks
        :param mask_size: the size of the mask
        :param mask_transform: transformations for the masks
        """
        self.img_dpath = img_dpath
        self.img_fnames = img_fnames
        self.img_transform = img_transform

        self.mask_encodings = mask_encodings
        self.mask_size = mask_size
        self.mask_transform = mask_transform

    def __getitem__(self, i):
        """
        Method that returns i-th image in the dataset
        :param i: index of the image
        :return: image with index i
        """
        seed = np.random.randint(2)
        # get the filename of the i-th image and form the path to it
        fname = self.img_fnames[i]
        fpath = os.path.join(self.img_dpath, fname)
        # open the image
        img = Image.open(fpath)
        # transform image
        if self.img_transform is not None:
            random.seed(seed)
            img = self.img_transform(img)
        # init the mask
        mask = np.zeros(self.mask_size, dtype=np.uint8)
        # if encodings exist, fill the mask with them
        if self.mask_encodings[fname][0] == self.mask_encodings[fname][0]:  # NaN doesn't equal to itself
            for encoding in self.mask_encodings[fname]:
                mask += rle_decode(encoding, self.mask_size)
        mask = np.clip(mask, 0, 1)
        # convert mask to Image instance
        mask = Image.fromarray(mask)
        # transform the mask
        random.seed(seed)
        mask = self.mask_transform(mask)
        return img, torch.from_numpy(np.array(mask, dtype=np.int64))

    def __len__(self):
        """
        Methods that returns the length of the dataset
        :return: the length of the dataset
        """
        return len(self.img_fnames)
