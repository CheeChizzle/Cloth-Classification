import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize
import torchvision
import zarr
import pickle

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True)

class ClothDataset(Dataset):
    classes = ['Dress', 'Jumpsuit', 'Skirt', 'Top', 'Trousers', 'Tshirt'] # list of classes/labels (y value, ground truth)
    def __init__(self, resolution, zarr_file='garmentnets_images.zarr', keys='train.pkl', use_single_view=False): # constructor which allows paramters to be passed through when an instance of the class is made
        # instance variables for image resolution, the actual zarr file, the flag to indicate the use of single view, and the cache list
        self.resolution = resolution
        self.zarr_file = zarr_file
        self.use_single_view = use_single_view
        self.cache_list = {}

        # using pickle to load file in as keys
        with open(keys, 'rb') as file:      
            self.keys = pickle.load(file)
        print(f'[ClothDataset] loaded {len(self.keys)} keys from {keys}')


        # calculating standardizaiton, equation is ((value - mean)/standard deviation)
        # after converging, we find that the mean and standard deviation of all rgb values are 53 and 45
        # self.rgb_mean = 53
        # self.rgb_std = 45
        # self.depth_mean = 0
        # rgbs = []
        # masks = []
        # for i, (category, instance) in enumerate(self.keys):
        #    if i > 200:
        #        break
        #    with zarr.open(f'{self.zarr_file}/{category}/samples/{instance}', mode='r') as z:
        #        rgb = torch.tensor(z['rgb'][:]).float() # shape: 4 x 1024 x 1024 x 3
        #        mask = torch.tensor(z['mask'][:]).float() # shape: 4 x 1024 x 1024 x 1
        #        img, class_ = trainset[i] # img shape: 3 x 32 x 32
        #        img = img.permute(1, 2, 0) # img shape: 32 x 32 x 3
        #        for view in range(4):
        #            rgb = rgb[view,:,:,:] # shape: 1024 x 1024 x 3
        #            rgb[(mask == 1)] = 0

               
               
        #self.rgb_mean =  torch.mean(torch.stack(rgbs))
        #self.depth_mean =  torch.mean(torch.stack(depths))
        #self.rgb_std = torch.std(torch.stack(rgbs))
        #self.depth_std = torch.std(torch.stack(depths))
        # print(self.rgb_mean, self.rgb_std)
        #print(self.depth_mean, self.depth_std)

        # creating a resize class to resize any given category to the specified resolution (when an instance of the class is created)
        self.resize = Resize(self.resolution)
    
    # this function acts as the len() function. it returns what len would return if it were called on an instance of this class
    def __len__(self):
        return len(self.keys)

    # implements what should happen when you access an index from an instance of the ClothDataset class
    def __getitem__(self, idx):

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

                mask = torch.tensor(z['mask'][:]).bool()
                mask[(mask == float('inf'))] = -1 #getting rid of infinity values
                
                # loading in background image to be used for mask
                img, _ = trainset[idx] # img shape: 3 x 32 x 32
                img = self.resize(img)  # img shape: 3 x 64 x 64
                

                
                # depth_normalized = (depth-self.depth_mean)/self.depth_std

                # if the flag that indicates using single view is true,
                if self.use_single_view:
                    rgb = rgb[0,:,:,:] # select one image view in the tensor. shape becomes: 1024 x 1024 x 3
                    # depth_normalized = depth_normalized[0,:,:,:]

                    # change the locations of the dimensions in rgb and resize to resolution size. shape becomes: 3 x 64 x 64
                    rgb = self.resize(rgb.permute(2,0,1)) 

                    # in mask, background is True and cloth is False
                    # in order to flip order, mask = 1-mask can be used
                    # rgb, img, and mask shape: 3 x 64 x 64
                    rgb[mask] = img[mask]

                    
                    
                    
                    # depth_normalized = self.resize(depth_normalized.permute(2,0,1))


                    # NOTE: images should be standardized
                    # standardization: turning dataset into a unit gaussian where mean is zero, std is one
                    # mean(), std(), ((x - mean)/std)
                    # compute mean and std of all rgb_normalized images in divide it as so
                else: # if multi view is being used (single view is false)
                    
                    # change the locations of the dimensions in rgb and resize to resolution size. shape becomes:  4 x 3 x 64 x 64
                    rgb = self.resize(rgb.permute(0,3, 1, 2))
                    # depth_normalized = self.resize(depth_normalized.permute(0,3, 1, 2))

                    # applying the CFAIR-10 image as background in each image view
                    for view in range(4):
                        rgb = rgb[view,:,:,:] # shape is now: 3 x 64 x 64
                        rgb[mask] = img[mask]
                        
                
                # implementing standardization with mean and standard deviation variables
                rgb_standardized = (rgb-torch.mean(rgb))/torch.std(rgb)


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
