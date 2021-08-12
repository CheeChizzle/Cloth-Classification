from torch.utils.data import Dataset
import torch
import zarr
import pickle

class ClothDataset(Dataset):
    classes = ['Dress', 'Jumpsuit', 'Skirt', 'Top', 'Trousers', 'Tshirt']
    def __init__(self, resolution, zarr_file='garmentnets_images.zarr', keys='train.pkl'):
        self.resolution = resolution
        self.zarr_file = zarr_file
        with open(keys, 'rb') as file:
            self.keys = pickle.load(file)
        print(f'[ClothDataset] loaded {len(self.keys)} keys from {keys}')

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        category, instance = self.keys[idx]
        with zarr.open(f'{self.zarr_file}/{category}/samples/{instance}', mode='r') as z:
            rgb = torch.tensor(z['rgb'][:]).float()
            depth = torch.tensor(z['depth'][:]).float()
            mask = torch.tensor(z['mask'][:]).short()
            return rgb[0]
            # , depth, mask, self.classes.index(category) 
