from torch.utils.data import DataLoader,Dataset
from skimage import io,transform
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from torch import tensor
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import sys
torch.set_default_tensor_type(torch.DoubleTensor)


##Dataset类
class mydataset(Dataset):
    def __init__(self,dir,transform=None):
        self.dir = dir
        self.transform = transform
        self.images = os.listdir(self.dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_index  = self.images[index]
        img_path = os.path.join(self.dir,image_index)
        img = np.load(img_path)
        img = np.expand_dims(img,0)
        label = int(img_path.split('_')[0][-1])


        return img,label

### CovdNet
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(

            nn.Conv2d(1,1,3),
            nn.ReLU(),
            nn.MaxPool2d(2,2)

        )
        self.fc = nn.Sequential(
            nn.Linear(16129,128),
            nn.Sigmoid(),
            nn.Linear(128,1)
    )
    def forward(self,img):
        feature =  self.conv(img)
      #  print(feature.shape)
        output = self.fc(feature.view(-1,16129))

        return output

def draw(list,str):
    x= range(1,11)
    y=list
    plt.plot(x,y,linestyle='--')
    plt.title(str)
    plt.show()

def train():
    num_epochs = 10
    loss_list = []
    train_acc_list = []
    test_acc_list = []
    data = mydataset('./trainset')
    dataloader = DataLoader(data,batch_size=128,shuffle=True)

    test_data = mydataset('./testset')
    test_dataloader = DataLoader(test_data,batch_size=128,shuffle=True)

    net = LeNet()
    optimizer = optim.Adam(net.parameters(),lr=0.01)
    loss_fun = nn.MSELoss()
    for epoch in range(1,num_epochs+1):
        train_l_sum,train_acc_sum,n,test_acc_sum=0.0,0.0,0,0.0
        for data,label in dataloader:

            data = data.double()
            label = label.double()
            y_hat = net(data)
            loss = loss_fun(y_hat.view(-1,1),label.view(label.shape[0],1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_l_sum += loss.item()
            train_acc_sum += (y_hat.round().reshape(label.shape) == label).sum().item()
            #print(y_hat.round().reshape(label.shape))
            n += label.shape[0]


        for test_data,test_label in test_dataloader:

            net.double()

            predict_y = net(test_data.double())
            test_acc_sum += (predict_y.round().view(-1,1) == test_label.view(-1,1)).sum().item()


        print('epoch %d ,loss : %.5f,train_acc=%.5f,test_acc=%.5f' % (epoch, loss.item(), train_acc_sum / n,test_acc_sum/n))
        loss_list.append(loss.item())
        train_acc_list.append(100*train_acc_sum/n)
        test_acc_list.append(100*test_acc_sum/n)
    draw(train_acc_list,'Train-accuracy')
    draw(test_acc_list,'Test-accuracy')


train()