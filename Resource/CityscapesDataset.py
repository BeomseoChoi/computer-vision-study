import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from PIL import Image
import glob

class CityscapesDataset(Dataset):
    def __init__(self, root_dir : Path, is_training : bool):
        self.root_dir : Path = Path(root_dir)
        self.isTraining = is_training
        self.x = []
        self.y = []

        self.load_data()

    def __len__(self):
        assert len(self.x) == len(self.y)
        return len(self.x)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x[idx]).permute(2, 0, 1)
        y = torch.FloatTensor(self.y[idx]).permute(2, 0, 1)
        return x,y

    def load_data(self):
        dir_name : str = "train" if self.isTraining else "val"
        filepaths : list[str] = glob.glob((self.root_dir / dir_name).__str__() + "/*.jpg")
        for filepath in filepaths:
            image : np.ndarray = np.array(Image.open(filepath))
            x = image[:, :int(image.shape[1] / 2)] / 255.
            y = image[:, int(image.shape[1] / 2):] / 255.
            
            self.x.append(x)
            self.y.append(y)

            