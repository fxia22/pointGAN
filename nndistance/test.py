import torch
import torch.nn as nn
from torch.autograd import Variable

from modules.nnd import NNDModule

dist =  NNDModule()

points1 = Variable(torch.rand(10,1000,3),requires_grad = True)
points2 = Variable(torch.rand(10,1500,3))

dist1, dist2 = dist(points1, points2)
print(dist1, dist2)

loss = torch.sum(dist1)
print(loss)
loss.backward()

print(points1.grad, points2.grad)