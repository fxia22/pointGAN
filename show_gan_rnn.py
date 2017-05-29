from __future__ import print_function
from show3d_balls import *
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets import PartDataset
from pointnet import PointGen, PointGenR
import torch.nn.functional as F
import matplotlib.pyplot as plt


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '',  help='model path')



opt = parser.parse_args()
print (opt)

gen = PointGenR()
gen.load_state_dict(torch.load(opt.model))

#sim_noise = Variable(torch.randn(5, 2, 20))
#
#sim_noises = Variable(torch.zeros(5, 15, 20))
#
#for i in range(15):
#    x = i/15.0
#    sim_noises[:,i,:] = sim_noise[:,0,:] * x + sim_noise[:,1,:] * (1-x)
#
#points = gen(sim_noises)
#point_np = points.transpose(2,1).data.numpy()

sim_noise = Variable(torch.randn(5, 6, 20))
sim_noises = Variable(torch.zeros(5, 30 * 5,20))

for j in range(5):
    for i in range(30):
        x = (1-i/30.0)
        sim_noises[:,i + 30 * j,:] = sim_noise[:,j,:] * x + sim_noise[:,(j+1) % 5,:] * (1-x)

        
points = gen(sim_noises)
point_np = points.transpose(2,1).data.numpy()
print(point_np.shape)

for i in range(150):
    print(i)
    frame = showpoints_frame(point_np[i])
    plt.imshow(frame)
    plt.axis('off') 
    plt.savefig('%s/%04d.png' %('out_rgan', i), bbox_inches='tight')
    plt.clf()




#showpoints(point_np)

#sim_noise = Variable(torch.randn(5, 1000, 20))
#points = gen(sim_noise)
#point_np = points.transpose(2,1).data.numpy()
#print(point_np.shape)
#choice = np.random.choice(2500, 2048, replace=False)
#print(point_np[:, choice, :].shape)


#showpoints(point_np)

#np.savez('rgan.npz', points = point_np[:, choice, :])

