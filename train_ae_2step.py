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
parser.add_argument('--outf', type=str, default='ae2',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--num_points', type=int, default = 2048,  help='number of points')

opt = parser.parse_args()
print (opt)

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Chair'], npoints = 2048, parts_also = True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Chair'], train = False, npoints = 2048, parts_also = True)
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


ae = PointNetAE(num_points = opt.num_points/4)


if opt.model != '':
    ae.load_state_dict(torch.load(opt.model))



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
classifier = PointNetCls(k = 2, num_points = opt.num_points)




def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

ae.apply(weights_init)
gen.apply(weights_init)
classifier.apply(weights_init)

ae.cuda()
gen.cuda()
classifier.cuda()


#noise = Variable(torch.rand(5,100)).cuda()
#gen(noise)


print(ae)
print(gen)
print(classifier)

optimizer = optim.Adagrad(ae.parameters(), lr = 0.001)
optimizerG = optim.Adagrad(gen.parameters(), lr = 0.001)
optimizerD = optim.Adagrad(classifier.parameters(), lr = 0.0002)



nnd = NNDModule()

num_batch = len(dataset)/opt.batchSize

for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()
        points, parts = data
           
        parts = torch.cat(parts, 0)
        #print(parts.size())
        
        pt = Variable(parts)

        bs = pt.size()[0]
        pt = pt.transpose(2,1)
        pt = pt.cuda()
        
        recon = ae(pt)
        
        pt = pt.transpose(2,1).contiguous()
        recon = recon.transpose(2,1).contiguous()
        
        dist1, dist2 = nnd(recon,pt)
        #print(recon.size(), points.size(), dist1.size())
        
        loss = torch.mean(dist1) + torch.mean(dist2)
        loss.backward()
        optimizer.step()
        
        #train G and D
        optimizerD.zero_grad()
        points = Variable(points)

        bs = points.size()[0]
        target = Variable(torch.from_numpy(np.ones(bs,).astype(np.int64))).cuda()
        points = points.transpose(2,1)
        points = points.cuda()
        #print(points.size())

        pred, trans = classifier(points)
        loss1 = F.nll_loss(pred, target)

        sim_noise = Variable(torch.randn(bs, 100)).cuda()
        fake, pos = gen(sim_noise)   
        fake = fake.contiguous()
        pos = pos.contiguous()
        fake_hidden = fake.view(-1,100)
        fake_gen = ae.decoder(fake_hidden)
        fake_gen = fake_gen.view(bs, 4, 3, opt.num_points / 4)
        pos = pos.view(bs, 4, 3, 1).repeat(1,1,1,opt.num_points / 4)
        
        fake_sample = pos + fake_gen
        fake_sample = fake_sample.transpose(2,1).contiguous().view(bs, 3, opt.num_points)
        
        #print(fake_sample.size())
        
        fake_target = Variable(torch.from_numpy(np.zeros(bs,).astype(np.int64))).cuda()
        pred2, trans2 = classifier(fake_sample)

        loss2 = F.nll_loss(pred2, fake_target)


        lossD = (loss1 + loss2)/2
        lossD.backward()
        #print(pred, target)

        optimizerD.step()
        
        
        optimizerG.zero_grad()
        
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
        
        pred, trans2 = classifier(fake_sample)
        target = Variable(torch.from_numpy(np.ones(bs,).astype(np.int64))).cuda()
        
        ##print(pred, target)
        lossG = F.nll_loss(pred, target)
        lossG.backward()
        optimizerG.step()
        
        
        
        print('[%d: %d/%d] train loss AE:%.4f lossG:%.4f lossD:%.4f' %(epoch, i, num_batch, loss.data[0], lossG.data[0], lossD.data[0]))

    torch.save(ae.state_dict(), '%s/model_ae_%d.pth' % (opt.outf, epoch))
    torch.save(classifier.state_dict(), '%s/modelD_%d.pth' % (opt.outf, epoch))
    torch.save(gen.state_dict(), '%s/modelG_%d.pth' % (opt.outf, epoch))