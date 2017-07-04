from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, num_points = 2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #x = self.mp1(x)
        #print(x.size())
        x,_ = torch.max(x, 2)
        #print(x.size())
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_points = 2500, global_feat = True, trans = True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points = num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.trans = trans


        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat
    def forward(self, x):
        batchsize = x.size()[0]
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans)
            x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        if self.trans:
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
        else:
            return x

class PointNetCls(nn.Module):
    def __init__(self, num_points = 2500, k = 2):
        super(PointNetCls, self).__init__()
        self.num_points = num_points
        self.feat = PointNetfeat(num_points, global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    def forward(self, x):
        x, trans = self.feat(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x), trans

class PointDecoder(nn.Module):
    def __init__(self, num_points = 2048, k = 2):
        super(PointDecoder, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, self.num_points * 3)
        self.th = nn.Tanh()
    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.th(self.fc5(x))
        x = x.view(batchsize, 3, self.num_points)
        return x


class PointNetAE(nn.Module):
    def __init__(self, num_points = 2048, k = 2):
        super(PointNetAE, self).__init__()
        self.num_points = num_points
        self.encoder = nn.Sequential(
        PointNetfeat(num_points, global_feat=True, trans = False),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Linear(256, 100),
        )

        self.decoder = PointDecoder(num_points)



    def forward(self, x):

        x = self.encoder(x)

        x = self.decoder(x)

        return x


class PointNetReg(nn.Module):
    def __init__(self, num_points = 2500, k = 1):
        super(PointNetReg, self).__init__()
        self.num_points = num_points
        self.feat = PointNetfeat(num_points, global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    def forward(self, x):
        x, trans = self.feat(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, trans



class PointNetReg2(nn.Module):
    def __init__(self, num_points = 500, k = 3):
        super(PointNetReg2, self).__init__()
        self.num_points = num_points
        self.feat = PointNetfeat(num_points, global_feat=True)
        self.fc1 = nn.Linear(1024 * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 100)
        self.fc4 = nn.Linear(100, k)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    def forward(self, x1, x2):
        x1, trans1 = self.feat(x1)
        x2, trans2 = self.feat(x2)

        x = torch.cat([x1,x2], 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = self.fc4(x)

        return x, trans1, trans2



class PointNetDenseCls(nn.Module):
    def __init__(self, num_points = 2500, k = 2):
        super(PointNetDenseCls, self).__init__()
        self.num_points = num_points
        self.k = k
        self.feat = PointNetfeat(num_points, global_feat=False)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)

    def forward(self, x):
        batchsize = x.size()[0]
        x, trans = self.feat(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k))
        x = x.view(batchsize, self.num_points, self.k)
        return x, trans


class PointGen(nn.Module):
    def __init__(self, num_points = 2500):
        super(PointGen, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, self.num_points * 3)

        self.th = nn.Tanh()
    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.th(self.fc4(x))
        x = x.view(batchsize, 3, self.num_points)
        return x

class PointGenComp(nn.Module):
    def __init__(self, num_points = 2500):
        super(PointGenComp, self).__init__()
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 500 * 3)
        self.encoder = PointNetfeat(num_points = 2000)
        self.th = nn.Tanh()
    def forward(self, x, noise):
        batchsize = x.size()[0]
        x, _ = self.encoder(x)
        x = torch.cat([x, noise], 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.th(self.fc4(x))
        x = x.view(batchsize, 3, 500)
        return x

class PointGenComp2(nn.Module):
    def __init__(self, num_points = 2500):
        super(PointGenComp2, self).__init__()
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 2500 * 3)
        self.encoder = PointNetfeat(num_points = 2000)
        self.th = nn.Tanh()
    def forward(self, x, noise):
        batchsize = x.size()[0]
        x, _ = self.encoder(x)
        x = torch.cat([x, noise], 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.th(self.fc4(x))
        x = x.view(batchsize, 3, 2500)
        return x


class PointGenR(nn.Module):
    def __init__(self, num_points = 2500):
        super(PointGenR, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 500 * 3)
        self.lstm = nn.LSTM(input_size = 20, hidden_size= 100, num_layers = 2)
        self.th = nn.Tanh()


    def forward(self, x):
        batchsize = x.size()[1]
        x, _ = self.lstm(x)
        x = x.view(-1,100)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.th(self.fc4(x))

        x = x.view(5, batchsize, 1500)

        x = x.transpose(1,0).contiguous()
        x = x.view(batchsize, 7500)

        x = x.view(batchsize, 3, 2500)
        return x

class PointGenR2(nn.Module):
    def __init__(self, num_points = 2500):
        super(PointGenR2, self).__init__()

        self.decoder = nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 500 * 3),
        nn.Tanh(),
        )

        self.lstmcell =   nn.LSTMCell(input_size = 100, hidden_size= 100)

        self.encoder = nn.Sequential(
        PointNetfeat(num_points = 500),
        )
        self.encoder2 = nn.Sequential(
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 100),
        )


    def forward(self, x):
        batchsize = x.size()[0]
        outs = []
        out = self.decoder(x)
        out = out.view(batchsize, 3, 500)
        outs.append(out)

        hx = Variable(torch.zeros(batchsize, 100))
        cx = Variable(torch.zeros(batchsize, 100))
        if x.is_cuda:
            hx = hx.cuda()
            cx = cx.cuda()

        for i in range(4):
            hd,_ = self.encoder(outs[-1])
            hd = self.encoder2(hd)
            hx, cx = self.lstmcell(hd, (hx, cx))

            out = self.decoder(hx)
            out = out.view(batchsize, 3, 500)
            outs.append(out)


        x = torch.cat(outs, 2)

        return x



class PointGenR3(nn.Module):
    def __init__(self, num_points = 2500):
        super(PointGenR3, self).__init__()

        def get_decoder():
            return nn.Sequential(
            nn.Linear(200, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 500 * 3),
            nn.Tanh(),
            )
        self.decoder = get_decoder()

        self.lstmcell =   nn.LSTMCell(input_size = 100, hidden_size= 100)

        self.encoder = nn.Sequential(
        PointNetfeat(num_points = 500),
        )
        self.encoder2 = nn.Sequential(
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 100),
        )


    def forward(self, x):
        batchsize = x.size()[0]

        hx = Variable(torch.zeros(batchsize, 100))
        cx = Variable(torch.zeros(batchsize, 100))

        outs = []

        if x.is_cuda:
            hx = hx.cuda()
            cx = cx.cuda()

        for i in range(5):
            if i == 0:
                hd = Variable(torch.zeros(batchsize, 100))
            else:
                hd,_ = self.encoder(torch.cat(outs, 2))
                hd = self.encoder2(hd)
            #print(hd.size())
            if x.is_cuda:
                hd = hd.cuda()

            hx, cx = self.lstmcell( hd, (hx, cx))

            out = self.decoder(torch.cat([hx,x[:,:,i]], 1))
            out = out.view(batchsize, 3, 500)
            outs.append(out)

        x = torch.cat(outs, 2)

        return x

class PointGenC(nn.Module):
    def __init__(self, num_points = 2500):
        super(PointGenC, self).__init__()
        self.conv1 = nn.ConvTranspose1d(100, 1024, 2,2,0)
        self.conv2 = nn.ConvTranspose1d(1024, 512, 5,5,0)
        self.conv3 = nn.ConvTranspose1d(512, 256, 5,5,0)
        self.conv4 = nn.ConvTranspose1d(256, 128, 2,2,0)
        self.conv5 = nn.ConvTranspose1d(128, 64, 5,5,0)
        self.conv6 = nn.ConvTranspose1d(64, 3, 5,5,0)

        self.bn1 = torch.nn.BatchNorm1d(1024)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.bn4 = torch.nn.BatchNorm1d(128)
        self.bn5 = torch.nn.BatchNorm1d(64)
        self.th = nn.Tanh()
    def forward(self, x):

        batchsize = x.size()[0]
        x = x.view(-1, 100, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        x = self.th(x)
        return x


class PointGenPSG(nn.Module):
    def __init__(self, num_points = 2048):
        super(PointGenPSG, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, self.num_points / 4 * 3 * 1)
        self.th = nn.Tanh()

        self.conv1 = nn.ConvTranspose2d(100,1024,(2,3))
        self.conv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.conv3 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.conv4= nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.conv5= nn.ConvTranspose2d(128, 3, 4, 2, 1)

        self.bn1 = torch.nn.BatchNorm2d(1024)
        self.bn2 = torch.nn.BatchNorm2d(512)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.bn5 = torch.nn.BatchNorm2d(3)



    def forward(self, x):
        batchsize = x.size()[0]

        x1 = x
        x2 = x

        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))
        x1 = self.th(self.fc4(x1))
        x1 = x1.view(batchsize, 3, self.num_points / 4 * 1)

        x2 = x2.view(-1, 100, 1, 1)
        x2 = F.relu((self.conv1(x2)))
        x2 = F.relu((self.conv2(x2)))
        x2 = F.relu((self.conv3(x2)))
        x2 = F.relu((self.conv4(x2)))
        x2 = self.th((self.conv5(x2)))

        x2 = x2.view(-1, 3, 32 * 48)
        #print(x1.size(), x2.size())

        return torch.cat([x1, x2], 2)

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())

    pointfeat = PointNetfeat(global_feat=True)
    out, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _ = seg(sim_data)
    print('seg', out.size())

    sim_data = Variable(torch.rand(32,3,500))
    pointreg = PointNetReg2()
    out, _, _ = pointreg(sim_data, sim_data)
    print('reg2', out.size())

