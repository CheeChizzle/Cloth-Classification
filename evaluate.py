import torch
# from statistics import mode
from tensorboardX import SummaryWriter
from network import loss_func

def evaluate(net, testloader):

    correct = 0
    testing_loss = 0.0
    with torch.no_grad():
        for batch in testloader:
            rgb, depth, mask, label = batch
            rgb, label = rgb.cuda(), label.cuda()
            # votes = []
            # for net in networks:
            output = net(rgb)

                # votes.append(output)

            # stacked = torch.stack(votes, dim=0) # size [num_networks, num_batchsize, num_classes]
            
            # avg_prob = torch.mean(stacked, dim=0) # size [num_batchsize, num_classes]

            pred_labels = torch.argmax(output, dim=-1) # size [num_batchsize]

            this_num_correct = (pred_labels == label).type(torch.int32).sum() # boolean tensor converted to integers. 1 corresponds to True and 0 to False

            loss = loss_func(output, label)
            testing_loss += loss.item()
            
            correct += this_num_correct
            
    
    accuracy = 100. * correct/len(testloader.dataset)
    return accuracy, correct, testing_loss

