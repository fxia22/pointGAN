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
from pointnet import PointNetCls, PointGen, PointNetReg
import torch.nn.functional as F



parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='wgan',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--clamp_lower', type=float, default=-0.02)
parser.add_argument('--clamp_upper', type=float, default=0.02)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')


opt = parser.parse_args()
print (opt)

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Chair'], classification = True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Chair'],classification = True, train = False)
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


classifier = PointNetReg()
gen = PointGen()


if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))
  
print(classifier)
print(gen)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        

classifier.apply(weights_init)
gen.apply(weights_init)

classifier.cuda()
gen.cuda()
    
optimizerD = optim.Adagrad(classifier.parameters(), lr = 0.001)
optimizerG = optim.Adagrad(gen.parameters(), lr = 0.001)


num_batch = len(dataset)/opt.batchSize
one = torch.FloatTensor([1]).cuda()
mone = one * -1


for epoch in range(opt.nepoch):
    data_iter = iter(dataloader)
    i = 0
    while i < len(dataloader):
        
        
        for diter in range(opt.Diters):
            for p in classifier.parameters():
                p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            optimizerD.zero_grad()
            data = data_iter.next()
            i += 1
            
            if i >= len(dataloader):
                break
            points, _ = data
            
            points = Variable(points)

            bs = points.size()[0]
            points = points.transpose(2,1) 
            points = points.cuda()
            #print(points.size())

            pred_real, trans = classifier(points)
            loss_real = torch.mean(pred_real) 
            
            sim_noise = Variable(torch.randn(bs, 100)).cuda()
            fake = gen(sim_noise)
            pred_fake, trans2 = classifier(fake)
            loss_fake = torch.mean(pred_fake)
            lossD = loss_real - loss_fake
            lossD.backward(one)
            #print(pred, target)
            optimizerD.step()
            print('[%d: %d/%d] train lossD: %f' %(epoch, i, num_batch, lossD.data[0]))
        
        optimizerG.zero_grad()
        sim_noise = Variable(torch.randn(bs, 100)).cuda()
        points = gen(sim_noise)
        pred, trans = classifier(points)
        #print(pred, target)
        
        lossG = torch.mean(pred)    
        lossG.backward(one)
        
        optimizerG.step()
        
        print('[%d: %d/%d] train lossD: %f lossG: %f' %(epoch, i, num_batch, lossD.data[0], lossG.data[0]))
       
    
    torch.save(classifier.state_dict(), '%s/modelD_%d.pth' % (opt.outf, epoch))
    torch.save(gen.state_dict(), '%s/modelG_%d.pth' % (opt.outf, epoch))