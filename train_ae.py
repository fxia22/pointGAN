from __future__ import print_function
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
from pointnet import PointNetCls, PointGen, PointGenC, PointNetAE
import torch.nn.functional as F

import sys
sys.path.insert(0, './nndistance')
from modules.nnd import NNDModule


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='ae',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--num_points', type=int, default = 2048,  help='number of points')

opt = parser.parse_args()
print (opt)

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Chair'], classification = True, npoints = 2048)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Chair'],classification = True, train = False, npoints = 2048)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))


cudnn.benchmark = True


print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)


try:
    os.makedirs(opt.outf)
except OSError:
    pass


ae = PointNetAE(num_points = opt.num_points)


if opt.model != '':
    ae.load_state_dict(torch.load(opt.model))

print(ae)



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

ae.apply(weights_init)

ae.cuda()

optimizer = optim.Adagrad(ae.parameters(), lr = 0.001)
nnd = NNDModule()

num_batch = len(dataset)/opt.batchSize

for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()
        points, _ = data
        points = Variable(points)

        bs = points.size()[0]
        points = points.transpose(2,1)
        points = points.cuda()
        
        gen = ae(points)
        
        points = points.transpose(2,1).contiguous()
        gen = gen.transpose(2,1).contiguous()
        
        dist1, dist2 = nnd(gen,points)
        #print(gen.size(), points.size(), dist1.size())
        
        loss = torch.mean(dist1) + torch.mean(dist2)
        loss.backward()
        optimizer.step()
        
        
        print('[%d: %d/%d] train loss %f' %(epoch, i, num_batch, loss.data[0]))

    torch.save(ae.state_dict(), '%s/model_ae_%d.pth' % (opt.outf, epoch))