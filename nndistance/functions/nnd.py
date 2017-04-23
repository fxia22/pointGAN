# functions/add.py
import torch
from torch.autograd import Function
from _ext import my_lib


class NNDFunction(Function):
    def forward(self, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()   
        self.xyz1 = xyz1
        self.xyz2 = xyz2
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)
        
        self.idx1 = torch.zeros(batchsize, n).type(torch.LongTensor)
        self.idx2 = torch.zeros(batchsize, m).type(torch.LongTensor)
        
        my_lib.nnd_forward(xyz1, xyz2, dist1, dist2, self.idx1, self.idx2)
        
        self.dist1 = dist1
        self.dist2 = dist2
        
        #print(batchsize, n, m)

        return dist1, dist2

    def backward(self, graddist1, graddist2):
        
        gradxyz1 = torch.zeros(self.xyz1.size())
        gradxyz2 = torch.zeros(self.xyz2.size())
        
        my_lib.nnd_backward(self.xyz1, self.xyz2, gradxyz1, gradxyz2, graddist1, graddist2, self.idx1, self.idx2)
        
        return gradxyz1, gradxyz2