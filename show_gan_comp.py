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
from pointnet import PointGen, PointGenC, PointGenComp
import torch.nn.functional as F
import matplotlib.pyplot as plt


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '',  help='model path')



opt = parser.parse_args()
print (opt)


dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Chair'], shape_comp = True)




gen = PointGenComp()
gen.load_state_dict(torch.load(opt.model))

ld = len(dataset)

idx = np.random.randint(ld)

print(ld, idx)

_,part = dataset[idx]

sim_noise = Variable(torch.randn(2, 1024))
sim_noises = Variable(torch.zeros(30,1024))
for i in range(30):
    x = i/30.0
    sim_noises[i] = sim_noise[0] * x + sim_noise[1] * (1-x)

part = Variable(part.view(1,2000,3).transpose(2,1)).repeat(30,1,1)

points = gen(part, sim_noises)
print(points.size(), part.size())
points = torch.cat([points, part], 2)

cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:,:3]

color = cmap[np.array([0] * 500 + [2] * 2000), :]

point_np = points.transpose(2,1).data.numpy()

showpoints(point_np, color)


