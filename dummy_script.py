from loader import ClothDataset
import torch
from argparse import ArgumentParser

# creates arguments 
parser = ArgumentParser()
parser.add_argument('--logdir', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--img_resolution', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--epochs', type=int, default=11)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--use_single_view', action='store_true') # store_truemakes use_single_view automatically false
parser.add_argument('--use_domain_randomization', action='store_true')
parser.add_argument('--load_ckpt',type=str,default=None) # for saving network
# add freeze resnet parameter
parser.add_argument('--freeze_resnet', action='store_true')
parser.add_argument('--num_networks', type=int, default=1)

args = parser.parse_args()

trainset = ClothDataset(args.img_resolution, keys="train.pkl", use_single_view=args.use_single_view, domain_randomization=args.use_domain_randomization)
testset = ClothDataset(args.img_resolution, keys="test.pkl", use_single_view=args.use_single_view, domain_randomization=args.use_domain_randomization)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)

from tqdm import tqdm
while True:
    for i, (rgb, depth, mask, label) in enumerate(tqdm(trainloader)):
        #print(rgb.shape)
        pass
    print('done')
