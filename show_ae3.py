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
parser.add_argument('--num_points', type=int, default = 2048,  help='model path')


opt = parser.parse_args()
print (opt)

ae = PointNetAE(num_points = 2048/4)
ae.load_state_dict(torch.load(opt.model + '_ae_40.pth'))



class PointCodeGen(nn.Module):
    def __init__(self, num_points = 2048):
        super(PointCodeGen, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, (100) * 4)


        self.fc5 = nn.Linear(100, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 64)
        self.fc8 = nn.Linear(64, (3) * 4)


        self.th = nn.Tanh()
    def forward(self, x):
        batchsize = x.size()[0]
        x1 = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))
        x1 = self.fc4(x1)

        code = x1.view(batchsize, 4, 100)

        x2 = F.relu(self.fc5(x))
        x2 = F.relu(self.fc6(x2))
        x2 = F.relu(self.fc7(x2))
        x2 = self.fc8(x2)

        offset = x2.view(batchsize, 4, 3)

        return code, offset




gen = PointCodeGen()
gen.load_state_dict(torch.load(opt.model + 'G_40.pth'))



dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Chair'], parts_also = True, npoints = 2048)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                          shuffle=True, num_workers=1)


ae.cuda()
gen.cuda()

bs = 32

sim_noise = Variable(torch.randn(bs, 100)).cuda()
fake, pos = gen(sim_noise)
fake = fake.contiguous()
pos = pos.contiguous()
fake_hidden = fake.view(-1,100)
fake_gen = ae.decoder(fake_hidden)
fake_gen = fake_gen.view(bs, 4, 3, opt.num_points / 4)
pos = pos.view(bs, 4, 3, 1).repeat(1,1,1,opt.num_points / 4)

fake_sample = pos + fake_gen
fake_sample = fake_sample.transpose(2,1)
#print(fake_sample.size())
fake_sample = fake_sample.contiguous().view(bs, 3, opt.num_points)

point_np = fake_sample.cpu().transpose(2,1).data.numpy()
print(point_np.shape)

showpoints(point_np)

#sim_noise = Variable(torch.randn(1000, 100))
#points = gen(sim_noise)
#point_np = points.transpose(2,1).data.numpy()
#print(point_np.shape)

#np.savez('gan.npz', points = point_np)
