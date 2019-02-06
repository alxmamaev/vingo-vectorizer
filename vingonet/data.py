import glob
from math import ceil
from random import choice
from . import utils
import cv2
import torch
from torch import nn
from torchvision.transforms import ToTensor
from albumentations import CenterCrop


class TrainDataloader:
    def __init__(self, data_path, augmentations,
                 image_size=(224, 244), batch_size=32,
                 hard_negative=False):
        self.image_size = image_size
        self.files = glob.glob(data_path + "/*.jpg")
        self.augmentations = augmentations
        self.to_tensor = ToTensor()
        self.curent_batch = 0
        self.batch_size = batch_size

    def load_image(self, image_path, augment=True):
        image = cv2.imread(image_path)

        crop_size = min(image.shape[:-1])
        image = CenterCrop(crop_size, crop_size)(image=image)["image"]
        image = cv2.resize(image, (600, 600))

        if augment:
            image = self.augmentations(image=image)["image"]

        image = cv2.resize(image, self.image_size)
        image = self.to_tensor(image)

        return image

    def get_triplet(self, indx):
        anchor_path = self.files[indx]
        negative_path = choice(self.files)

        anchor = utils.load_image(anchor_path)
        positive = utils.load_image(anchor_path, augmentations=self.augmentations)
        negative = utils.load_image(negative_path, augmentations=self.augmentations)

        return anchor, positive, negative

    def get_batch(self):
        start = self.curent_batch * self.batch_size
        end = (self.curent_batch+1) * self.batch_size
        end = min(end, len(self.files)-1)

        anchor = []
        positive = []
        negative = []

        for i in range(start, end):
            a, p, n = self.get_triplet(i)
            anchor.append(a)
            positive.append(p)
            negative.append(n)

        anchor, positive, negative = torch.stack(anchor), torch.stack(positive), torch.stack(negative)
        return anchor, positive, negative

    def __len__(self):
        return ceil(len(self.files) / self.batch_size)

    def __iter__(self):
        self.curent_batch = 0
        return self

    def __next__(self):
        return self.get_batch()
