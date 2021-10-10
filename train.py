# imports
# import evaluate
from argparse import ArgumentParser
from network import SingleViewNet, MultiViewNet, SingleViewResNet, MultiViewResNet, seed_all, loss_func
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
from tensorboardX import SummaryWriter
from evaluate import evaluate
 
# TODO: evaluate.py file, similar but takes in a checkpoint and returns confidence, accuracy, loss, (on both training and testing set)
# TODO: visualze (in notebook) image instances of network with highest loss and image instances of network with lowest loss
# values correspond to name of network class. key will be used to access them 
networks ={
    'singleviewnet': SingleViewNet,
    'multiviewnet': MultiViewNet,
    'singleviewnetresnet': SingleViewResNet,
    'multiviewnetresnet': MultiViewResNet
}

# creates arguments 
parser = ArgumentParser()
parser.add_argument('--logdir', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--img_resolution', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--epochs', type=int, default=11)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--use_single_view', action='store_true') # store_truemakes use_single_view automatically false
parser.add_argument('--use_domain_randomization', action='store_true')
parser.add_argument('--load_ckpt',type=str,default=None) # for saving network
parser.add_argument('--arch', choices=list(networks.keys()), default='singleviewnet')
# add freeze resnet parameter
parser.add_argument('--freeze_resnet', action='store_true')
parser.add_argument('--num_networks', type=int, default=1)

args = parser.parse_args()
os.mkdir(args.logdir)
# setting up
seed_all(args.seed)

if args.arch == "singleviewnetresnet" or args.arch == "multiviewnetresnet":
    net = networks[args.arch](freeze_layers = args.freeze_resnet).cuda()
else:
    net = networks[args.arch]().cuda()


opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

trainset = ClothDataset(args.img_resolution, keys="train.pkl", use_single_view=args.use_single_view, domain_randomization=args.use_domain_randomization)
testset = ClothDataset(args.img_resolution, keys="test.pkl", use_single_view=args.use_single_view, domain_randomization=args.use_domain_randomization)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)


if args.load_ckpt is not None:
    ckpt = torch.load(args.load_ckpt)
    net.load_state_dict(ckpt['network'])
    opt.load_state_dict(ckpt['optimizer'])

start = time()
max_ckpt_accuracy = 0
training_step = 0
epoch_step = 0

for epoch in range(args.epochs):
    # print("Experiment:", args.logdir, "Epoch #:", epoch+1)
   
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

    # img = torch.tensor()
    # img.shape = C W H
    # crop: 
    # img = img[:,10:-10,10:-10]

    #TODO: Research tensorboard
    
    logger = SummaryWriter(args.logdir)
    
    # ssh -L 6002:127.0.0.1:6002 chichi@128.59.23.32
    # tensorboard --logdir <expdir> --port 6002 --host 0.0.0.0
    # localhost:6002

    if args.load_ckpt is None:
        # training
        training_loss = 0.0
        net.train()
        for i, (rgb, depth, mask, label) in enumerate(tqdm(trainloader, total=len(trainloader), smoothing=0.01, dynamic_ncols=True)):
            # multiview: B x 4 x 3 x W x H
            # B x 3 x W x H
            # singleview: B x W x H
            #rgb = torch.tensor(rgb).to(dtype = torch.float)
            # rgb = rgb[0,:,:,:,:]
            
            # print min, max, and mean of batch of rgb images
            opt.zero_grad()
            output = net(rgb.cuda())
            loss = loss_func(output, label.cuda())

            loss.backward()

            # research other experiences/tefchniques with resnet50
            # Check gradient norm
            # list of gradient norms
            grad_norms = []
            for p in list(filter(lambda p: p.grad is not None, net.parameters())):
                grad_norms.append(p.grad.detach().data.norm(2).cpu())
            # print min, mean, and max of gradient norm list
            # if len(grad_norms) != 0:
            # print("GRADIENT NORM STATS | Min:", min(grad_norms), "Mean:", (sum(grad_norms)/len(grad_norms)), "Max:", max(grad_norms))

            logger.add_scalar('mean_gradnorm', (sum(grad_norms)/len(grad_norms)), training_step)
            logger.add_histogram('gradnorms', grad_norms, training_step)
            
            # TODO: implement the following:
            # log loss (done),  average training loss (done), average testing loss (done), total training accuracy (done), and accuracy for different cloth types at ever step
            # log images for each step (2 images: one with the highest correct predictions and one with the least)
            # use matplotlip or pillow/pil (python image library) to output image and display network's loss, ground truth/label, and network's prediction
            # TODO: Run one full epoch to see if implementation is running correctly

            # print(grad_norms)

            opt.step()

            # print statistics every once in a while
            training_loss += loss.item()
            logger.add_scalar("loss", training_loss, training_step)
            training_step+=1
            
            # if i % 200 == 199:
            #     print('[EPOCH %d, BATCH %5d] LOSS: %.3f' %
            #           (epoch + 1, i + 1, training_loss / 200))
            #     
            #     training_loss = 0.0
        print("Train set: Average training loss:", training_loss/len(trainloader))
        logger.add_scalar('mean_training_loss', (training_loss/len(trainloader)), epoch_step)
        logger.add_histogram('training_loss', training_loss, epoch_step)
        
    
    networks = []
    for i in range(args.num_networks):
        seed_all(i+1)
        new_net = net.cuda()
        networks.append(new_net)

    accuracy, correct = evaluate(networks, testloader, logger)   
    # test_loss /= (len(testloader.dataset)/args.batch_size)
    print("\n Test set: Accuracy:", accuracy)
    # logger.add_scalar('mean_testing_loss', test_loss, epoch_step)
    # logger.add_histogram('testing_loss', test_loss, epoch_step)
    

    current_ckpt_accuracy = 100. * correct / len(testloader.dataset)
    logger.add_scalar('accuracy', current_ckpt_accuracy, epoch_step)
    
    if current_ckpt_accuracy > max_ckpt_accuracy:
        max_ckpt_accuracy = current_ckpt_accuracy
        # save the network
        torch.save({
            'network': net.state_dict(),
            'optimizer': opt.state_dict()
        },f'{args.logdir}/ckpt_{epoch}.pth')

    epoch_step+=1

finish = time()
print('Computation took ', float(finish - start), ' seconds')