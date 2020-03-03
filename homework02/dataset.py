from torch.utils.data import Dataset
from os.path import join
from PIL import Image


class ImgsDataset(Dataset):
    def __init__(self, filenames, targets=None, base_dir='.', transform=None):
        self.filenames = filenames 
        self.targets = targets
        self.base_dir = base_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(join(self.base_dir, self.filenames[idx])).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        assert tuple(img.shape) == (3, 64, 64), img.shape
        if self.targets:
            return img, self.targets[idx]
        
        return img
    
