# imports
from argparse import ArgumentParser
from network import SingleViewNet, MultiViewNet, ResNetModel, seed_all, loss_func
from loader import ClothDataset
import torch
# import numpy as np
# import torchvision
import torch.optim as optim
import torch.nn.functional as F
from threading import Thread
from time import time
from tqdm import tqdm
import os
networks ={
    'singleviewnet': SingleViewNet,
    'multiviewnet': MultiViewNet,
    'singleviewnetresnet': ResNetModel
}

parser = ArgumentParser()
parser.add_argument('--logdir', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--use_single_view', action='store_true') # store_truemakes use_single_view automatically false
parser.add_argument('--load_ckpt',type=str,default=None) # for saving network
parser.add_argument('--arch', choices=list(networks.keys()), default='singleviewnet')

args = parser.parse_args()
os.mkdir(args.logdir)
# setting up
seed_all(args.seed)

if args.arch != "resnet":
    net = networks[args.arch]().cuda()
else:
    net = networks[args.arch].cuda()


opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

trainset = ClothDataset(64, keys="train.pkl", use_single_view=args.use_single_view)
testset = ClothDataset(64, keys="test.pkl", use_single_view=args.use_single_view)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)


if args.load_ckpt is not None:
    ckpt = torch.load(args.load_ckpt)
    net.load_state_dict(ckpt['network'])
    opt.laod_state_dict(ckpt['optimizer'])
 
 #

start = time()
for epoch in range(args.epochs):
    # B C W H (4 dimension)
    # B V C W H (5 dimension)

    # Single View
    
    # input = loader_iter.next()
    # input.shape == B C W H
    # net(input).shape == B K

    """
    t=0 [     #       #       #       #      # ]
    t=1 [   net(#)  net(#)  net(#)  net(#)  net(#)]
    t=2 [    |       |        |       |      | ]
    """


    # B X
    """

    t=0 [     #       #       #       #      # ]
    t=1 [     #                                ]
    t=2 [   net(#)  ]
    t=3 [    |      ]
    """
    # NOTE: 
    # - Always think about the shape
    # - Understand everyline in the code (e.g.: flatten, linear constructor arguments)

    # Multi View
    # Input: B V C H W
    # Net: V C H W ==> K
    # def forward(images):
    #    # images.shape B V C H W 
    #    # CNNs expect 4 dimensional imputs
    #    # image_feat_extractor == CNN
    #    # views = []
    #    # len(views) == 4
    #    # views[0].shape == B C H W
    #    # first_view = images[:,0,...] (B C H W)
    #    # second_view = images[:,1,...]
    #    # ...

    # Output: B K


    # training
    running_loss = 0.0
    for i, (rgb, depth, mask, label) in enumerate(tqdm(trainloader, total=len(trainloader), smoothing=0.01, dynamic_ncols=True)):
        # multiview: B x 4 x 3 x W x H
        # B x 3 x W x H
        #rgb = torch.tensor(rgb).to(dtype = torch.float)
        # rgb = rgb[0,:,:,:,:]
        opt.zero_grad()
        output = net(rgb.cuda())
        loss = loss_func(output, label.cuda())

        loss.backward()
        opt.step()

        # print statistics every once in a while
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[EPOCH %d, BATCH %5d] LOSS: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    
    # testing
    test_loss = 0
    correct = 0
    for batch in testloader:
        # taking first datapoint in trainloader iterable object
        rgb, depth, mask, label = batch

        rgb, label = rgb.cuda(), label.cuda()

        output = net(rgb) # perform a forward pass to get the network's predictions
        test_loss += loss_func(output, label)
        _, predicted = torch.max(output.data, 1)

        correct += predicted.eq(label.data.view_as(predicted)).long().cpu().sum()
    
    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    
    # save the network
    torch.save({
        'network': net.state_dict(),
        'optimizer': opt.state_dict()
    },f'{args.logdir}/ckpt_{epoch}.pth')


finish = time()
print('computation took ', float(finish - start), ' seconds')
