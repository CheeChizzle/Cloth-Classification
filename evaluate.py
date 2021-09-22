import torch
# import numpy as np
import torchvision
import numpy as np
import random
from network import seed_all, loss_func

def ensemble_learning_run(checkpoint, net_type, num_of_nets, testloader):


    networks = []
    for i in range(num_of_nets):
        seed_all(i+1)

        ckpt = torch.load(checkpoint)
        
        new_net = net_type.load_state_dict(ckpt['network']).cuda()
        opt = opt.load_state_dict(ckpt['optimizer'])
    
    ensemble_learning_evaluate(networks, testloader)
    
    

def ensemble_learning_evaluate(networks, testloader):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in testloader:
            rgb, depth, mask, label = batch
            rgb, label = rgb.cuda(), label.cuda()
            # for net in networks:

            scores = [net(rgb) for net in networks]

            votes = [torch.max(score.data, 1) for max, score in scores]

        num = len(np.unique(votes))
        if num == len(votes):
            predicted =  random.choice(votes)
        predicted = torch.mode(torch.tensor(votes))[0]

        if predicted == label:
                correct += 1
    
    accuracy = correct/len(testloader.dataset)
    
         
    # multiple sv models for singleview classification
    return confidence, accuracy, loss