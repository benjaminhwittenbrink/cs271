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


class SatelliteImageMetadataDataset(Dataset):

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
        country = img_dhsid[:2]
        country_int = int(str(ord(country[0])) + str(ord(country[1])))

        return img, label, country_int 


class SatelliteImageMosaiksDataset(Dataset):

    def __init__(self, root, csv_path, outcome, mosaiks_csv_path, loader):

        # root and image paths to load images 
        self.root = root 
        self.image_paths = os.listdir(root)
        self.img_loader = loader

        # data to load outcome 
        self.data = pd.read_csv(csv_path + ".csv")
        self.outcome = outcome
        self.outcome_dict = dict(zip(self.data["DHSID"], self.data[outcome]))

        # to load mosaiks_features 
        self.mosaiks_data = pd.read_csv(mosaiks_csv_path + ".csv")
        self.mosaiks_features = [" ." + str(i+1) for i in range(3999)]
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_filepath = self.root + "/" + self.image_paths[idx]
        img = self.img_loader(image_filepath)

        # everything before extension 
        img_dhsid = self.image_paths[idx].split('.')[0]
        label = self.outcome_dict[img_dhsid]
        mosaiks_features = self.mosaiks_data.loc[self.mosaiks_data['DHSID'] == img_dhsid][self.mosaiks_features]
        mosaiks_features = np.array(mosaiks_features.iloc[0])

        return img, label, mosaiks_features 
