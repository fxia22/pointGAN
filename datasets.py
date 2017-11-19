from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import progressbar
import sys
import torchvision.transforms as transforms
import argparse
import json


class PartDataset(data.Dataset):
    def __init__(self, root, npoints = 2500, classification = False, class_choice = None, train = True, parts_also = False, shape_comp = False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.parts_also = parts_also
        self.shape_comp = shape_comp
        
        self.classification = classification
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        #print(self.cat)
        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
            
        self.meta = {}
        for item in self.cat:
            #print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], 'points')
            dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            #print(dir_point, dir_seg)
            fns = sorted(os.listdir(dir_point))
            if train:
                fns = fns[:int(len(fns) * 0.9)]
            else:
                fns = fns[int(len(fns) * 0.9):]
                
            #print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0]) 
                self.meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))
        
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))
            
         
        self.classes = dict(zip(self.cat, range(len(self.cat))))  
        self.num_seg_classes = 0
        if not self.classification:
            for i in range(len(self.datapath)/50):
                l = len(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
                if l > self.num_seg_classes:
                    self.num_seg_classes = l
        #print(self.num_seg_classes)
        
        
    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        #print(point_set.shape, seg.shape)
        
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]
        seg = seg[choice]
        parts = []
        if self.parts_also:
            for j in np.unique(seg):
            
                part = point_set[seg == j]
                choice2 = np.random.choice(part.shape[0], self.npoints/4, replace=True)
                part = part[choice2, :]
                #print(part.shape)
                part = part - np.expand_dims(np.mean(part, axis = 0), 0)
                part = torch.from_numpy(part)
                parts.append(part)
        
        
        if self.shape_comp:
            num_seg = len(np.unique(seg))
            j = np.random.randint(num_seg) + 1
            #print(len(point_set))
            incomp = point_set[seg != j]
            #print(len(incomp))
            
            choice2 = np.random.choice(incomp.shape[0], 4 * self.npoints/5, replace=True)
            incomp = incomp[choice2, :]
            #print(part.shape)
            incomp = torch.from_numpy(incomp)
            
            
        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0)
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        dist = np.expand_dims(np.expand_dims(dist, 0), 1)
        
        if not self.parts_also:
            point_set = point_set/dist
            
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        
        if self.shape_comp:
            return point_set, incomp
        elif self.parts_also:
            return point_set, parts
        
        elif self.classification:

            return point_set, cls
        else:
            return point_set, seg
        
         
        
        
    def __len__(self):
        return len(self.datapath)


class PartPosDataset(data.Dataset):
    def __init__(self, root, npoints = 2500, classification = False, class_choice = None, train = True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}

        self.classification = classification
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        #print(self.cat)
        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
            
        self.meta = {}
        for item in self.cat:
            #print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], 'points')
            dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            #print(dir_point, dir_seg)
            fns = sorted(os.listdir(dir_point))
            if train:
                fns = fns[:int(len(fns) * 0.9)]
            else:
                fns = fns[int(len(fns) * 0.9):]
                
            #print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0]) 
                self.meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))
        
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))
            
         
        self.classes = dict(zip(self.cat, range(len(self.cat))))  
        self.num_seg_classes = 0
        if not self.classification:
            for i in range(len(self.datapath)/50):
                l = len(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
                if l > self.num_seg_classes:
                    self.num_seg_classes = l
        #print(self.num_seg_classes)
        
        
    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        
        #print(point_set.shape, seg.shape)
        
        choice = np.random.choice(len(seg), self.npoints, replace=True)
      
            
        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0)
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        dist = np.expand_dims(np.expand_dims(dist, 0), 1)
        point_set = point_set/dist
            
        num_seg = len(np.unique(seg))
        parts_np = np.zeros((num_seg, 500, 3))
        seg_id = np.zeros((num_seg, 1))
        centers = np.zeros((num_seg, 3))
        
        
        for i,sid in enumerate(np.unique(seg)):
            part = point_set[seg == sid, :]
            center = np.mean(part, axis = 0)
            part = part - np.expand_dims(np.mean(part, axis = 0), 0)
            
            #print(center)
            centers[i, :] = center
            choice = np.random.choice(part.shape[0], 500, replace=True)
            parts_np[i, :, :] = part[choice, :]
            seg_id[i] = sid
           
        #from IPython import embed; embed()

        point_set = torch.from_numpy(parts_np.astype(np.float32))
        seg_id = torch.from_numpy(seg_id.astype(np.int64))
        centers = torch.from_numpy(centers.astype(np.float32))
 
        
        return point_set, centers, seg_id

    
        
    def __len__(self):
        return len(self.datapath)
    
    
    
if __name__ == '__main__':
    print('test')
    d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Chair'])
    print(len(d))
    ps, seg = d[0]
    print(ps.size(), ps.type(), seg.size(),seg.type())
    print(torch.mean(ps, 0))
    d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True)
    print(len(d))
    ps, cls = d[0]
    print(ps.size(), ps.type(), cls.size(),cls.type())
    
    d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0',  parts_also = True)
    print(len(d))
    ps, cls = d[0]
    print(ps.size(), ps.type(), cls[0].size(),cls[0].type())
    
    
    d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0',  shape_comp = True)
    print(len(d))
    ps, inc = d[0]
    print(ps.size(), ps.type(), inc.size(),inc.type())
    
    d = PartPosDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Chair'])
    print(len(d))
    ps, centers, seg = d[0]
    print(ps.size(), ps.type(), seg.size(),seg.type())
    
    