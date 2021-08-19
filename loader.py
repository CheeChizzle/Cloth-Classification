import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize
import zarr
import pickle

class ClothDataset(Dataset):
    classes = ['Dress', 'Jumpsuit', 'Skirt', 'Top', 'Trousers', 'Tshirt']
    def __init__(self, resolution, zarr_file='garmentnets_images.zarr', keys='train.pkl', use_single_view=False, use_rgbd = False):
        self.resolution = resolution
        self.zarr_file = zarr_file
        self.use_single_view = use_single_view
        self.use_rgbd = use_rgbd
        with open(keys, 'rb') as file:
            self.keys = pickle.load(file)
        print(f'[ClothDataset] loaded {len(self.keys)} keys from {keys}')
        rgbs = []
        depths = []
        self.rgb_mean = 53
        self.rgb_std = 45
        # self.depth_mean = 0
        #for i, (category, instance) in enumerate(self.keys):
        #    if i > 200:
        #        break
        #    with zarr.open(f'{self.zarr_file}/{category}/samples/{instance}', mode='r') as z:
        #        rgbs.append(torch.tensor(z['rgb'][:]).float())
        #        depth = torch.tensor(z['depth'][:]).float()
        #        depth[depth == float("inf")] = -1
        #        depths.append(depth)
        #self.rgb_mean =  torch.mean(torch.stack(rgbs))
        #self.depth_mean =  torch.mean(torch.stack(depths))
        #self.rgb_std = torch.std(torch.stack(rgbs))
        #self.depth_std = torch.std(torch.stack(depths))
        print(self.rgb_mean, self.rgb_std)
        #print(self.depth_mean, self.depth_std)

        self.resize = Resize(self.resolution)
        

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        category, instance = self.keys[idx]
        with zarr.open(f'{self.zarr_file}/{category}/samples/{instance}', mode='r') as z:
            rgb = torch.tensor(z['rgb'][:]).float()
            rgb[(rgb == float('inf'))] = -1 #getting rid of infinity values

            # depth = torch.tensor(z['depth'][:]).float()
            # depth[(depth == float('inf'))] = -1 #getting rid of infinity values

            # mask = torch.tensor(z['mask'][:]).bool()
            # mask[(mask == float('inf'))] = -1 #getting rid of infinity values
            
            rgb_normalized = (rgb-self.rgb_mean)/self.rgb_std
            # depth_normalized = (depth-self.depth_mean)/self.depth_std

            if self.use_single_view:
                rgb_normalized = rgb_normalized[0,:,:,:]
                # depth_normalized = depth_normalized[0,:,:,:]

                rgb_normalized = self.resize(rgb_normalized.permute(2,0,1))
                # depth_normalized = self.resize(depth_normalized.permute(2,0,1))


                # NOTE: images should be standardized
                # standardization: turning dataset into a unit gaussian where mean is zero, std is one
                # mean(), std(), ((x - mean)/std)
                # compute mean and std of all rgb_normalized images in divide it as so
            else:
                rgb_normalized = self.resize(rgb_normalized.permute(0,3, 1, 2))
                # depth_normalized = self.resize(depth_normalized.permute(0,3, 1, 2))
            if self.use_rgbd:
                pass # will implement later
            
            # depth = self.resize(depth)
            return rgb_normalized, dict(), dict(), self.classes.index(category)
            #return rgb_normalized, depth_normalized, mask, self.classes.index(category)

# resources
# 1. request to use
# 2. use
# 3. release

# resource example: file
# 1. request to use: file = open('hello_world.txt','r')
# 2. use: txt = file.read(); print(txt)
# 3. file.close()

# resource example: zarr
# 1. file = zarr.open('mydataset.zarr', mode='r')
# 2. rgb = file['rgb']
# 3. file.close()

# index: arr[0] ==> __getitem__
# len: len(arr) ==> __len__
# with [resource]: __enter__, __exit__

# ✅ import torch vision transform, get transforms (resize function) to resize images to preferred resolution. Try making image size 64x64 and go from there 
# ✅ worker threads in dataloader (pytorch) 
# Change all infinity values in tensors to -1
# ✅ depth[depth == inf] = -1
