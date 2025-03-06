import os
import pandas as pd
import skimage.draw
import torch.utils.data
import numpy as np
import torch.utils.data
from pycocotools.coco import COCO
import pickle
import torchvision.transforms.v2 as transforms
import torchvision.io
from PIL import Image


class PartImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, mode: str = 'train', get_masks: bool = False, image_size: int = 224,
                 evaluate: bool = False):
        """
        PartImageNet dataset
        Parameters
        ----------
        data_path: str
            Directory containing the 'train_train', 'train_test', and 'test' folders
        mode: str
            Whether to use the training or validation split
        get_masks: bool
            Whether to return the ground truth masks
        image_size: int
            Size of the images
        evaluate: bool
            Set to true to evaluate parts (disables transforms such as normalization, crop, etc.)
        """
        self.mode = mode
        self.data_path = data_path
        self.get_masks = get_masks
        dataset = pd.read_csv(data_path + "/" + "newdset.txt", sep='\t', names=["index", "test", "label", "class", "filename"])
        if mode == "train":
            self.dataset = dataset.loc[dataset['test'] == 0]
            self.transform = self.get_transforms(image_size, evaluate)[0]
        elif mode == "val":
            self.dataset = dataset.loc[dataset['test'] == 1]
            self.transform = self.get_transforms(image_size, evaluate)[1]
        elif mode == "test":
            self.dataset = dataset.loc[dataset['test'] == 1]
            self.transform = self.get_transforms(image_size, evaluate)[1]
        annFile = os.path.join(data_path, f"train.json")

        coco = COCO(annFile)
        self.coco = coco

    def getmasks(self, i):
        idx = self.dataset.iloc[i]['index']
        idx = int(idx)
        coco = self.coco
        img = coco.loadImgs(idx)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        cat_ids = [ann['category_id'] for ann in anns]
        polygons = []
        for ann in anns:
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                polygons.append(poly)
        for cat, p in zip(cat_ids, polygons):
            mask = skimage.draw.polygon2mask((img['width'], img['height']), p)
            try:
                mask_tensor[cat] += torch.FloatTensor(mask)
            except NameError:
                mask_tensor = torch.zeros(size=(40, mask.shape[-2], mask.shape[-1]))
                mask_tensor[cat] += torch.FloatTensor(mask)
        try:
            mask_tensor = torch.where(mask_tensor > 0.1, 1, 0).permute(0, 2, 1)
            return mask_tensor
        except UnboundLocalError:
            # if an image has no ground truth parts
            return None

    def __len__(self):
        return len(self.dataset['index'])

    def __getitem__(self, idx):
        curr_row = self.dataset.iloc[idx]
        folder = curr_row['class']
        imgname = curr_row['filename']
        
        if self.mode == 'train':
            path = f"{self.data_path}/train_train/{folder}/{imgname}"
        elif self.mode == 'test':
            path = f"{self.data_path}/train_test/{folder}/{imgname}"

        if os.path.isfile(path) and os.path.getsize(path) > 0:
            im = torchvision.io.read_image(path, torchvision.io.ImageReadMode.RGB)
            label = curr_row['label']
            im = self.transform(im)
    
            if not self.get_masks:
                return im, label
    
            mask = self.getmasks(idx)
            if mask == None:
                mask = torch.zeros(size=(40, im.shape[-2], im.shape[-1]))
            mask = transforms.Resize(size=(im.shape[-2], im.shape[-1]),
                    interpolation=transforms.InterpolationMode.NEAREST)(mask)
            return im, label, mask
        else:
            return None, None, None

    @staticmethod
    def get_transforms(image_size: int, evaluate: bool = False):
        if not evaluate:
            train_transforms = transforms.Compose([
                transforms.Resize(size=image_size, antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1),
                transforms.RandomAffine(degrees=90, translate=(0.2, 0.2), scale=(0.8, 1.2)),
                transforms.RandomCrop(image_size),
                transforms.ToDtype(torch.float32, scale=True)
            ])
            test_transforms = transforms.Compose([
                transforms.Resize(size=image_size, antialias=True),
                transforms.CenterCrop(size=image_size),
                transforms.ToDtype(torch.float32, scale=True)
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.Resize(size=image_size, antialias=True),
                transforms.ToDtype(torch.float32, scale=True)
            ])
            test_transforms = transforms.Compose([
                transforms.Resize(size=image_size, antialias=True),
                transforms.ToDtype(torch.float32, scale=True)
            ])
        return train_transforms, test_transforms