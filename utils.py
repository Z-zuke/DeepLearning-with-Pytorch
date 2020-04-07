#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
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

