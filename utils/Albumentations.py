from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize,Rotate, Cutout, CoarseDropout,PadIfNeeded
from albumentations.pytorch import ToTensor
import numpy as np


class album_compose_train:
    mean1=[0.5, 0.5, 0.5]
    std1=[0.5, 0.5, 0.5]
    def __init__(self,mean,std):
        self.mean1=mean
        self.std1=std
        self.albumentation_transforms = Compose([
            #Rotate((-7.0, 7.0)),
            PadIfNeeded(min_height=36, min_width=36),
            RandomCrop(32, 32),
            HorizontalFlip(),
            Cutout(num_holes=4),
            #CoarseDropout(),
            #HorizontalFlip(),
            #RandomCrop(),
            Normalize(
                mean=self.mean1,
                std=self.std1
            ), ToTensor()])

    def __call__(self, img):
        img = np.array(img)
        img = self.albumentation_transforms(image=img)['image']
        return img
class album_compose_test:
    mean1=[0.5, 0.5, 0.5]
    std1=[0.5, 0.5, 0.5]
    def __init__(self,mean,std):
        self.mean1=mean
        self.std1=std
        self.albumentation_transforms = Compose([
            Normalize(
                mean=self.mean1,
                std=self.std1
            ), ToTensor()])

    def __call__(self, img):
        img = np.array(img)
        img = self.albumentation_transforms(image=img)['image']
        return img