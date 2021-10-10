import torch
# from statistics import mode
from tensorboardX import SummaryWriter

testing_step = 0
def evaluate(networks, testloader, logger):
    
    testing_step = 0
    correct = 0
    with torch.no_grad():
        for batch in testloader:
            rgb, depth, mask, label = batch
            rgb, label = rgb.cuda(), label.cuda()
            votes = []
            for net in networks:
                output = net(rgb)

                votes.append(output)

            stacked = torch.stack(votes, dim=0) # size [num_networks, num_batchsize, num_classes]
            
            avg_prob = torch.mean(stacked, dim=0) # size [num_batchsize, num_classes]

            pred_labels = torch.argmax(avg_prob, dim=-1) # size [num_batchsize]

            this_num_correct = (pred_labels == label).type(torch.int32).sum() # boolean tensor converted to integers. 1 corresponds to True and 0 to False
            correct += this_num_correct
            print()
            logger.add_scalar("num_correct", correct, testing_step)
            testing_step+=1
    
    accuracy = 100. * correct/len(testloader.dataset)

    print("accuracy:", accuracy)
    print("correct:", correct)
    return accuracy, correct

