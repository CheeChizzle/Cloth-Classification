# imports
from loader import ClothDataset
import matplotlib.pyplot as plt
# import matplotlib
import torch
# import numpy as np
import torchvision
# from network import hyperparameters, Net
# matplotlib.use('MacOSX')

trainset = ClothDataset(64, keys="train.pkl", use_single_view=True)
# rgb, depth, mask, label = trainset[0]
# print(rgb.shape)

# datapoint = trainset[0]
# type(datapoint) == tuple, len(datapoint) == 4
# multiview_rgb = datapoint[0]
# type(multiview_rgb) == torch.tensor, multiview_rgb.shape == 4x1024x1024x3
# rgb = multiview_rgb[0]
# rgb.shape 1024x1024x3
# rgb/255
# [0, 255]

# print(rgb.max(),rgb.min(),rgb.max())

trainloader = torch.utils.data.DataLoader(trainset, batch_size=2,
                                        shuffle=True)

for batch in trainloader:
    rgb, depth, mask, label = batch
    print(rgb.shape)
exit()
# turning trainloader into iterable object
dataiter = iter(trainloader)

# taking first datapoint in trainloader iterable object
rgb, depth, mask, label = dataiter.next()

# rgb shape: 2 x 4 x 1024 x 1024 x 3
# depth shape: 2 x 4 x 1024 x 1024 x 1
# mask shape: 2 x 4 x 1024 x 1024 x 1
# label shape: 2

# single_view_rgb shape: 2 x 1024 x 1024 x 3. Corresponds to --> (B x H x W x C)

# rgb: B x H x W x C
# images = torch.movedim(rgb, 3, 1)
for img in rgb:
    # img: H x W x C
    img = (trainset.rgb_std * img + trainset.rgb_mean).long()
    plt.imshow(img.numpy())
    # [0,255]
    plt.show()
images = rgb.permute(0,3,1,2)
# 0 1 2 3
# V H W C

# B C H W
# 0 3 1 2

# print(images.shape)
# view = rgb[0,:, 0, 0, 0]
# print(view.shape)
# needs tensor of shape ([B x C x H x W])
img_grid = torchvision.utils.make_grid(images)



# loss_func = nn.CrossEntropyLoss() # Cross entropy loss

# optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=1e-3)

# def matplotlib_imshow(img, one_channel=False):
#     if one_channel:
#         img = img.mean(dim=0)
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     if one_channel:
#         plt.imshow(npimg, cmap="Greys")
#     else:
#         plt.imshow(np.transpose(npimg, (1, 2, 0)))

# matplotlib_imshow(img_grid, one_channel=False)

# matplotlib_imshow(img_grid, one_channel=True)
# print(dataset[0])