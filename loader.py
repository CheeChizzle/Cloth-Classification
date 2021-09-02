import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize
import torchvision.transforms as transforms
import torchvision
import zarr
import pickle
import random

transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform = transform)


class ClothDataset(Dataset):
    classes = ['Dress', 'Jumpsuit', 'Skirt', 'Top', 'Trousers', 'Tshirt'] # list of classes/labels (y value, ground truth)
    def __init__(self, resolution, zarr_file='garmentnets_images.zarr', keys='train.pkl', use_single_view=False, domain_randomization = False): # constructor which allows paramters to be passed through when an instance of the class is made
        # instance variables for image resolution, the actual zarr file, the flag to indicate the use of single view, and the cache list
        self.resolution = resolution
        self.zarr_file = zarr_file
        self.use_single_view = use_single_view
        self.cache_list = {}
        self.domain_randomization = domain_randomization
        # creating a resize class to resize any given category to the specified resolution (when an instance of the class is created)
        self.resize = Resize(self.resolution)
        self.rgb_mean = 0
        self.rgb_std = 0

        # using pickle to load file in as keys
        with open(keys, 'rb') as file:      
            self.keys = pickle.load(file)
        print(f'[ClothDataset] loaded {len(self.keys)} keys from {keys}')


        # calculating standardizaiton, equation is ((value - mean)/standard deviation)
        # after converging, we find that the mean and standard deviation of all rgb values are 53 and 45. With the CFAIR-10 images, they are 47 and 43
        if self.domain_randomization:
            self.rgb_mean = 0.2
            self.rgb_std = 0.18
        else:
            self.rgb_mean = 53
            self.rgb_std = 45
        # rgbs = []
        # for i, (category, instance) in enumerate(self.keys):
        #     if i > 200:
        #         break
        #     with zarr.open(f'{self.zarr_file}/{category}/samples/{instance}', mode='r') as z:
        #         main_rgb = torch.tensor(z['rgb'][:]).float() # shape: 4 x 1024 x 1024 x 3
        #         main_rgb[(main_rgb == float('inf'))] = -1
        #         # main_rgb = main_rgb/255
        #         mask = torch.tensor(z['mask'][:]).bool() # shape: 4 x 1024 x 1024 x 1
        #         mask[(mask == float('inf'))] = -1

        #         main_rgb = self.resize(main_rgb.permute(0, 3, 1, 2)) # rgb shape: 4 x 3 x 256 x 256
        #         mask = self.resize(mask.permute(0, 3, 1, 2)) # mask shape: 4 x 1 x 256 x 256
        #         mask = torch.squeeze(mask) # mask shape: 4 x 256 x 256

        #         for view in range(4): 
        #             view_rgb = main_rgb[view,:,:,:] # rgb shape: 3 x 256 x 256
        #             view_mask = mask[view,:,:] # mask shape: 256 x 256
        #             view_rgb = view_rgb/255

        #             rand_idx = random.randrange(len(trainset))
        #             # loading in random cfair-10 background image to be used for mask
        #             img, _ = trainset[rand_idx] # img shape: 3 x 32 x 32
        #             img = self.resize(img)  # img shape: 3 x 256 x 256
                    
        #             view_rgb[:,view_mask] = img[:,view_mask]
        #             main_rgb[view,:,:,:] = view_rgb
        #         rgbs.append(main_rgb)
            
        # self.rgb_mean =  torch.mean(torch.stack(rgbs))
        # self.rgb_std = torch.std(torch.stack(rgbs))
        # print("Mean:",self.rgb_mean, "Standard deviation",self.rgb_std)

    
    # this function acts as the len() function. it returns what len would return if it were called on an instance of this class
    def __len__(self):
        return len(self.keys)
    
    def get_random_img(self):
        rand_idx = random.randrange(len(trainset))
        # loading in background image to be used for mask
        img, _ = trainset[rand_idx] # img shape: 3 x 32 x 32 
        img = self.resize(img)  # img shape: 3 x 256 x 256
        return img
    
    # implements what should happen when you access an index from an instance of the ClothDataset class
    def __getitem__(self, idx):
        # print(len(self.cache_list)) # check lenght of cache list to understand limit while using 256+ resolution
        # the keys variable is a list. each value in keys contains a tuple of a category (eg dress) and an instance (name of file that contains the actual image)
        category, instance = self.keys[idx]
        # conditional statement to use cache. purpose is to make running faster/less expensive by resizing image only once (not each time training is run)
        if str(idx) not in self.cache_list: 
            # with keyword allows us to use resourse and release without using file.open() and file.close(). z is at top of file hierarchy, z[category] is next, z[category][samples][instance] is next and lastly z[category][samples][instance][depth/mask/rgb]. with the category and instance variables, we are able to access z right down to the instance and can now format the rgb, depth, and mask values inside of it.
            with zarr.open(f'{self.zarr_file}/{category}/samples/{instance}', mode='r') as z:

                # turning rgb matrix into a torch tensor then turning every unit in tensor to a float
                rgb = torch.tensor(z['rgb'][:]).float() # shape: 4 x 1024 x 1024 x 3

                rgb[(rgb == float('inf'))] = -1 # mask technique is used to get rid of infinity values
                

                # depth = torch.tensor(z['depth'][:]).float()
                # depth[(depth == float('inf'))] = -1 #getting rid of infinity values

                mask = torch.tensor(z['mask'][:]).bool() # shape: 4 x 1024 x 1024 x 1
                mask[(mask == float('inf'))] = -1 #getting rid of infinity values

                mask = torch.squeeze(mask) # shape: 4 x 1024 x 1024
                # resize to resolution size. shape becomes: 4 x 256 x 256
                mask = self.resize(mask)
                
                

                # if the flag that indicates using single view is true,
                if self.use_single_view:
                    rgb = rgb[0,:,:,:] # select one image view in the tensor. shape becomes: 1024 x 1024 x 3

                    # change the locations of the dimensions in rgb and resize to resolution size. shape becomes: 3 x 256 x 256
                    rgb = self.resize(rgb.permute(2,0,1))

                    if self.domain_randomization:
                        mask = mask[0,:,:] # select one image view in the tensor. shape becomes: 256 x 256
                        rgb = rgb/255
                        img = self.get_random_img()
                        # in mask, background is True and cloth is False
                        # in order to flip order, mask = ~mask can be used
                        # rgb and img shape: 3 x 256 x 256
                        # mask shape: 256 x 256
                        rgb[:, ~mask] = img[:, mask]

                    
                    
                    
                    # depth_normalized = self.resize(depth_normalized.permute(2,0,1))


                    # NOTE: images should be standardized
                    # standardization: turning dataset into a unit gaussian where mean is zero, std is one
                    # mean(), std(), ((x - mean)/std)
                    # compute mean and std of all rgb_normalized images in divide it as so
                else: # if multi view is being used (use_single_view is false)
                    
                    # change the locations of the dimensions in rgb and resize to resolution size. shape becomes:  4 x 3 x 256 x 256
                    rgb = self.resize(rgb.permute(0,3, 1, 2))
                    # depth_normalized = self.resize(depth_normalized.permute(0,3, 1, 2))

                    if self.domain_randomization:
                        rgb = rgb/255
                        # applying a random CFAIR-10 image as background in each image view
                        for view in range(4):
                            img = self.get_random_img()
                            view_rgb = rgb[view,:,:,:] # shape: 3 x 256 x 256
                            view_mask = mask[view,:,:] # shape: 256 x 256

                            view_rgb[:, ~view_mask] = img[:, ~view_mask]
                            rgb[view,:,:,:] = view_rgb
                        
                
                # implementing standardization with mean and standard deviation variables
                rgb_standardized = (rgb - self.rgb_mean)/self.rgb_std


                # resized image is now added to the cache_list (dictionary) as a value with the index being its key
                self.cache_list[str(idx)] = rgb_standardized 
              
            
        return self.cache_list[str(idx)], dict(), dict(), self.classes.index(category)
        # return rgb_normalized, depth_normalized, mask, self.classes.index(category)

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
