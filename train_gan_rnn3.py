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
from pointnet import PointNetCls, PointGen, PointGenC, PointGenR3
import torch.nn.functional as F



parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='gan',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')

opt = parser.parse_args()
print (opt)

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Chair'], classification = True, parts_also = True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Chair'],classification = True, train = False, parts_also = True)
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


classifier = PointNetCls(k = 2)
classifier2 = PointNetCls(num_points = 500, k = 2)

gen = PointGenR3()


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
classifier2.apply(weights_init)
gen.apply(weights_init)

classifier.cuda()
classifier2.cuda()
gen.cuda()

optimizerD = optim.Adagrad(classifier.parameters(), lr = 0.001)
optimizerD2 = optim.Adagrad(classifier2.parameters(), lr = 0.001)
optimizerG = optim.Adagrad(gen.parameters(), lr = 0.001)


num_batch = len(dataset)/opt.batchSize

for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):
        optimizerD.zero_grad()
        optimizerD2.zero_grad()
        points, part = data
        points = Variable(points)
        part = Variable(part)


        bs = points.size()[0]
        target = Variable(torch.from_numpy(np.ones(bs,).astype(np.int64))).cuda()
        points = points.transpose(2,1)
        points = points.cuda()
        #print(points.size())
        pred, trans = classifier(points)
        loss1 = F.nll_loss(pred, target)

        part = part.transpose(2,1)
        part = part.cuda()
        pred, trans = classifier2(part)
        loss21 = F.nll_loss(pred, target)


        sim_noise = Variable(torch.randn(bs, 100,5)).cuda()
        fake = gen(sim_noise)
        fake_target = Variable(torch.from_numpy(np.zeros(bs,).astype(np.int64))).cuda()
        pred2, trans2 = classifier(fake)
        loss2 = F.nll_loss(pred2, fake_target)

        #print(fake.size())
        pred2, trans2 = classifier2(fake[:,:,:500])
        loss22 = F.nll_loss(pred2, fake_target)


        lossD = (loss1 + loss2)/2
        #lossD.backward()
        #print(pred, target)

        lossD2 = (loss21 + loss22)/2

        (lossD2 + lossD).backward()

        optimizerD.step()
        optimizerD2.step()


        optimizerG.zero_grad()
        sim_noise = Variable(torch.randn(bs, 100,5)).cuda()
        points = gen(sim_noise)
        pred, trans = classifier(points)
        target = Variable(torch.from_numpy(np.ones(bs,).astype(np.int64))).cuda()

        pred2, trans2 = classifier2(points[:,:,:500])


        #print(pred, target)
        lossG1 = F.nll_loss(pred, target)
        lossG2 = F.nll_loss(pred2, target)
        lossG = lossG1 + lossG2
        lossG.backward()
        optimizerG.step()

        print('[%d: %d/%d] train lossD: %f lossD2 %f lossG: %f lossG2: %f' %(epoch, i, num_batch, lossD.data[0], lossD2.data[0], lossG1.data[0], lossG2.data[0]))


    torch.save(classifier.state_dict(), '%s/modelD_%d.pth' % (opt.outf, epoch))
    torch.save(gen.state_dict(), '%s/modelG_%d.pth' % (opt.outf, epoch))
