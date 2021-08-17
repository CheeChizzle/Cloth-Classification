import zarr
import numpy as np
import pickle
import tqdm
import torch
from torchvision.transforms import Resize

z = zarr.open('garmentnets_images.zarr', mode='r')
resized_z = zarr.open('garmentnets_resized_images.zarr', mode='a')

keys = {}
"""
z 
 |
 | __ Dress
        |
        | _ samples
                |_ 31234214_dress_213412
                |_52342453_dress_34324234
 |
 |
 |___ Skirt 
 |
  ....

z 
 |
 | __ Dress
 |
 |
 |___ Skirt 
 |
  ....
"""
resize = Resize(64)
for category in z:
    category_group = resized_z.create_group(category)
    sample_group = category_group.create_group("samples")
    for instance in tqdm.tqdm(z[category]['samples'], desc=category):
        instance_group = sample_group.create_group(instance)
        rgb = torch.tensor(z[category]['samples'][instance]['rgb'][:]).float()
        # depth = torch.tensor(z[category]['samples'][instance]['depth'][:]).float()
        # mask = torch.tensor(z[category]['samples'][instance]['mask'][:]).bool()

        rgb_resized = resize(rgb)
        # depth_resized = resize(depth)

        # mask_resized = resize(mask)

        instance_group["rgb"] = rgb_resized.numpy()
        # instance_group["depth"] = depth_resized.numpy()
        # instance_group["mask"] = mask_resized.numpy()
        # instance_z = z[category]['samples'][instance]

z.close()
resized_z.close()