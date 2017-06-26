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
from pointnet import PointGen, PointGenC, PointNetAE
import torch.nn.functional as F
import matplotlib.pyplot as plt


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '',  help='model path')

opt = parser.parse_args()
print (opt)

ae = PointNetAE(num_points = 2048)
ae.load_state_dict(torch.load(opt.model))

dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Chair'], classification = True, npoints = 2048)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                          shuffle=True, num_workers=1)


ae.cuda()

i,data = enumerate(dataloader, 0).next()
points, _ = data
points = Variable(points)

bs = points.size()[0]
points = points.transpose(2,1)
points = points.cuda()
        
gen = ae(points)
point_np = gen.transpose(2,1).cpu().data.numpy()

    
#showpoints(points.transpose(2,1).cpu().data.numpy())
showpoints(point_np)

#sim_noise = Variable(torch.randn(1000, 100))
#points = gen(sim_noise)
#point_np = points.transpose(2,1).data.numpy()
#print(point_np.shape)

#np.savez('gan.npz', points = point_np)
