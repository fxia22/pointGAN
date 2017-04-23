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
from pointnet import PointGen, PointGenR3
import torch.nn.functional as F
import matplotlib.pyplot as plt


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '',  help='model path')



opt = parser.parse_args()
print (opt)

gen = PointGenR3()
gen.load_state_dict(torch.load(opt.model))

sim_noise = Variable(torch.randn(2, 100,5))

sim_noises = Variable(torch.zeros(30,100,5))

for i in range(30):
    x = i/30.0
    sim_noises[i] = sim_noise[0] * x + sim_noise[1] * (1-x)

points = gen(sim_noises)
point_np = points.transpose(2,1).data.numpy()

cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:,:3]

color = cmap[np.repeat(np.array([1,2,3,4,5]), 500), :]

showpoints(point_np, color)

