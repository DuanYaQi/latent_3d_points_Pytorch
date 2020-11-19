import torch
import torch.utils.data as data
import os
import json
import numpy as np
from plyfile import PlyData


class ShapeNetDataset(data.Dataset):
    def __init__(self,samples_dir):
        self.samples_dir = samples_dir
        self.samples_paths = os.listdir(samples_dir)
        

    def __getitem__(self, index):
        fn = self.samples_dir + '/' + self.samples_paths[index]
        point_set = PlyData.read(fn)
        point_set = point_set['vertex']
        point_set= np.vstack([point_set['x'], point_set['y'], point_set['z']]).T

        point_set = torch.from_numpy(point_set)
        return point_set


    def __len__(self):
        return len(self.samples_paths)
