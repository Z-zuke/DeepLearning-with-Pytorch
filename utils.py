#!/usr/bin/env python
# coding: utf-8

# In[1]:
import torch
import torchvision
from torch import nn,optim
from matplotlib import pyplot as plt


# In[2]:
def plot_curve(data):
    fig=plt.figure()
    plt.plot(range(len(data)),data, color='blue')
    plt.legend(['value'],loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()


# In[3]:
def plot_image(img, label, name):
    fig=plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(img[i][0]*0.3081+0.1307, cmap='gray', interpolation='none')
        plt.title('{}: {}'.format(name,label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


# In[4]:
def one_hot(label,depth=10):
    out=torch.zeros(label.size(0),depth)
    idx=torch.LongTensor(label).view(-1,1)
    out.scatter_(dim=1,index=idx,value=1)
    return out


# In[5]:
class Lenet5(nn.Module):
    
    def __init__(self):
        super(Lenet5,self).__init__()
        
        self.conv_unit=nn.Sequential(
            # x: [b,3,32,32] => [b,16,28,28]
            nn.Conv2d(3,16,kernel_size=5,stride=1,padding=0),
            # x: [b,16,28,28] => [b,16,14,14]
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            # x: [b,16,14,14] => [b,32,10,10]
            nn.Conv2d(16,32,kernel_size=5,stride=1,padding=0),
            # x: [b,32,10,10] => [b,32,5,5]
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        )
        # fully connect unit
        self.fc_unit=nn.Sequential(
            nn.Linear(32*5*5,32),
            nn.ReLU(),
            nn.Linear(32,10)
        )
        
    def forward(self,x):
        batchsz=x.size(0)
        # [b,3,32,32] => [b,32,5,5]
        x=self.conv_unit(x)
        # [b,32,5,5] => [b,16*5*5] flatten
        x=x.view(batchsz,-1)
        # [b,32*5*5] => [b,10]
        logits=self.fc_unit(x)
        
        return logits