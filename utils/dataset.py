<<<<<<< HEAD
import torch.utils.data as data
import os
import json
=======
import torch
import torch.utils.data as data
import os
import json
import numpy as np
from plyfile import PlyData

>>>>>>> 88c700e218d7e1d6f2d7fb92e8e4ff59a056d1cd

class ShapeNetDataset(data.Dataset):
    def __init__(self,samples_dir):
        self.samples_dir = samples_dir
        self.samples_paths = os.listdir(samples_dir)
<<<<<<< HEAD
       

    def __getitem__(self, index):
        fn = self.samples_paths[index]
        point_set = np.loadtxt(fn[1]).astype(np.float32)

        #resample
        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale
=======
        

    def __getitem__(self, index):
        fn = self.samples_dir + '/' + self.samples_paths[index]
        point_set = PlyData.read(fn)
        point_set = point_set['vertex']
        point_set= np.vstack([point_set['x'], point_set['y'], point_set['z']]).T
>>>>>>> 88c700e218d7e1d6f2d7fb92e8e4ff59a056d1cd

        point_set = torch.from_numpy(point_set)
        return point_set


    def __len__(self):
<<<<<<< HEAD
        return len(self.datapath)
=======
        return len(self.samples_paths)
>>>>>>> 88c700e218d7e1d6f2d7fb92e8e4ff59a056d1cd
