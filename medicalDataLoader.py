from __future__ import print_function, division
import os
import torch
import pandas as pd
from matplotlib import pyplot as plt
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps
from random import random, randint

# Ignore warnings
import warnings

import pdb

from imageDataGenerator import savePNG

warnings.filterwarnings("ignore")


def make_dataset(root, mode):
    assert mode in ['train','val', 'test']
    items = []

    if mode == 'train':
        train_img_path = os.path.join(root, 'train', 'Img')#creer le path /root/train/Img
        train_mask_path = os.path.join(root, 'train', 'GT')

        images = os.listdir(train_img_path)#le noms de fichiers de toutes les images
        labels = os.listdir(train_mask_path)

        images.sort()#les mets dans l'ordre
        labels.sort()

        for it_im, it_gt in zip(images, labels):#item liste -> ('chemin/image/spécifique', 'chemin/mask/correspondant')
            item = (os.path.join(train_img_path, it_im), os.path.join(train_mask_path, it_gt))
            items.append(item)


    elif mode == 'val':
        val_img_path = os.path.join(root, 'val', 'Img')
        val_mask_path = os.path.join(root, 'val', 'GT')

        images = os.listdir(val_img_path)
        labels = os.listdir(val_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (os.path.join(val_img_path, it_im), os.path.join(val_mask_path, it_gt))
            items.append(item)
    else:
        test_img_path = os.path.join(root, 'test', 'Img')
        test_mask_path = os.path.join(root, 'test', 'GT')

        images = os.listdir(test_img_path)
        labels = os.listdir(test_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (os.path.join(test_img_path, it_im), os.path.join(test_mask_path, it_gt))
            items.append(item)

    return items


class MedicalImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mode, root_dir, transform=None, mask_transform=None, augment=False, equalize=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir#chemin de base
        self.transform = transform #met en tensor (fonction)
        self.mask_transform = mask_transform#idem
        self.imgs = make_dataset(root_dir, mode)
        self.augmentation = augment#si besoin d'augmenter le dataset
        self.equalize = equalize#si besoin de equalize
        self.mode = mode#train / val

    def __len__(self):#retourne la longeur du dataset
        return len(self.imgs)

    def augment(self, img, mask):#operations sur l'image, augmente le nombre de données (aléatoire) 
        if random() > 0.5:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)
        if random() > 0.5:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
        if random() > 0.5:
            angle = random() * 60 - 30
            img = img.rotate(angle)
            mask = mask.rotate(angle)
        return img, mask

    def __getitem__(self, index):#retourne liste avec image, mask et chemin de l'image
        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path)
        mask = Image.open(mask_path).convert('L')#convertie en grayscale, 1 channel, entre 0 et 1

        if self.equalize:# ? non linea mapping uniform distribution of grayscale values
            img = ImageOps.equalize(img)

        if self.augmentation:#augmente nombre de donnée, à faire plus tard
            img, mask = self.augment(img, mask)
            # torchvision.transforms.AutoAugmentPolicy(value) à tester
            # torchvision.transforms.RandAugment(

        if self.transform: # met en tensor
            img = self.transform(img)
            mask = self.mask_transform(mask)
            # savePNG(img, 'Data/val/Prd/', 'test')

        return [img, mask, img_path]