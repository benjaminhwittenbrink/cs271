import os 
import pandas as pd 
import numpy as np 
from torch.utils.data import Dataset

class SatelliteImageDataset(Dataset):

    def __init__(self, root, csv_path, outcome, loader):

        self.root = root 
        self.image_paths = os.listdir(root)
        self.data = pd.read_csv(csv_path + ".csv")
        self.outcome = outcome
        self.outcome_dict = dict(zip(self.data["DHSID"], self.data[outcome]))
        self.img_loader = loader

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_filepath = self.root + "/" + self.image_paths[idx]
        img = self.img_loader(image_filepath)

        # everything before extension 
        img_dhsid = self.image_paths[idx].split('.')[0]
        label = self.outcome_dict[img_dhsid]

        return img, label
