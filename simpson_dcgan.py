import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data 
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

dataroot = "./simpsons_dataset"
batch_size = 1
dataset = dset.ImageFolder(root = dataroot,transform=transforms.Compose([
		transforms.RandomResizedCrop(224), 
		transforms.ToTensor()
		]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size ,shuffle=False)

means = np.array([0.0,0.0,0.0]).astype(float)
std = np.array([0.0,0.0,0.0]).astype(float)

for data in dataloader:
	for i in range(3):
		#print(data[0].size())
		means[i] += data[0][:,i,:,:].mean()
		std[i] += data[0][:,i,:,:].std()
	print(means, std)
print("Done")
print("Mean: ", means/len(dataset))
print("Std: ",std/len(dataset))


'''
real_batch = next(iter(dataloader))
print(real_batch[0][0].size())
#plt.imshow(np.transpose(vutils.make_grid(real_batch[0]),(1,2,0)))
print(torch.transpose(real_batch[0][0],0,2).size())
plt.imshow(torch.transpose(real_batch[0][1],0,2).numpy())
plt.show()
'''

#Total 20932