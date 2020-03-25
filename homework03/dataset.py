from torch.utils.data import Dataset
from os import listdir, path
from PIL import Image
import numpy as np


class ImgDataset(Dataset):
    def __init__(self, img_dir, mask_dir='', transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.imgs = listdir(img_dir)
        self.masks = listdir(mask_dir) if mask_dir else []
    
    def __len__(self):
        return len(self.imgs)

    def _get_img(self, img_file, img_dir):
        img = Image.open(path.join(img_dir, img_file))
        if self.transform:
            img = self.transform(img)

        return img
    
    def __getitem__(self, idx):
        img = Image.open(path.join(self.img_dir, self.imgs[idx]))

        if self.masks:
            mask = Image.open(path.join(self.mask_dir, self.masks[idx]))

#             if self.transform:
#                 stacked = Image.fromarray(np.concatenate([np.asarray(img), np.asarray(mask)[..., None]], axis=-1))

#                 stacked = self.transform(stacked)
#                 img, mask = stacked[:-1, ...], stacked[-1].long()                 
            img = self.transform(img)
            mask = self.transform(mask)
            return img, mask
        
        if self.transform:
            img = self.transform(img)

        return img