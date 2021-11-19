# load in your pretrained network
from network import MultiViewResNet
import pickle
import torch
import seaborn as sns
import matplotlib.pyplot as plt

classes = ['Dress', 'Jumpsuit', 'Skirt', 'Top', 'Trousers', 'Tshirt']
# create network
net = MultiViewResNet()
# torch load

ckpt = torch.load("/Users/brownscholar/Desktop/cloth-classification/finetuned_domainrand.pth", map_location=torch.device('cpu'))
net.load_state_dict(ckpt['network'])
data = pickle.load(open('/Users/brownscholar/Desktop/cloth-classification/domain-randomized-testset.pkl','rb'))
nodr =data['not-domain-randomized_multiview']

count = 0
net.eval()
for (rgb, _, _, label) in nodr:
    if count == 72 or count == 13 or count == 93 or count == 66:
        print("clothing item number", count)
        print("real clothing item", classes[label])
        

        rgb = torch.unsqueeze(rgb, 0)
        score = net(rgb)
        score = torch.squeeze(score)
        distribution = torch.nn.functional.softmax(score)

        sns.set_style('whitegrid')
        sns.barplot(y=distribution.detach().numpy(),x=['Dress', 'Jumpsuit', 'Skirt', 'Top', 'Trousers', 'Tshirt'], color='cornflowerblue').set(yticklabels=[])
        plt.show()
    count+=1