import torch
# import numpy as np
import torchvision
import numpy as np
import random
from network import seed_all, loss_func
# from statistics import mode
from tensorboardX import SummaryWriter

testing_step = 0
def evaluate(net_type, num_of_nets, testloader, logger):


    networks = []
    for i in range(num_of_nets):
        seed_all(i+1)
        
        new_net = net_type
        new_net = new_net.cuda()
        networks.append(new_net)
    
    ensemble_learning_evaluate(networks, testloader, logger)
    
    

def ensemble_learning_evaluate(networks, testloader, logger):
    testing_step = 0
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in testloader:
            rgb, depth, mask, label = batch
            rgb, label = rgb.cuda(), label.cuda()
            votes = []
            for net in networks:
                output = net(rgb)
                
                max, max_index = torch.max(output.data, 1)
                votes.append(max_index)

            num = len(np.unique(votes))
            if num == len(votes):
                predicted =  random.choice(votes)
            predicted = torch.mode(torch.tensor(votes))[0]

            test_loss += loss_func(predicted, label)
            if predicted == label:
                    correct += 1
            logger.add_scalar("num_correct", correct, testing_step)
            testing_step+=1
    
    accuracy = 100. * correct/len(testloader.dataset)

    
    return None, accuracy, test_loss, correct

