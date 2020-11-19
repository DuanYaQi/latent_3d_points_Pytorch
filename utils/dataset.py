import torch
import torch.utils.data as data
import os
import numpy as np
import warnings

from plyfile import PlyData

class ShapeNetDataset(data.Dataset):
    def __init__(self, samples_dir, sample_num):
        self.samples_dir = samples_dir
        self.samples_paths = os.listdir(samples_dir)
        if sample_num < len(self.samples_paths):
            self.samples_paths = self.samples_paths[0:sample_num]
        else:
            warnings.warn('Sample_num Overflow.')

    def __getitem__(self, index):
        fn = self.samples_dir + '/' + self.samples_paths[index]
        point_set = PlyData.read(fn)
        point_set = point_set['vertex']
        point_set= np.vstack([point_set['x'], point_set['y'], point_set['z']]).T

        point_set = torch.from_numpy(point_set)
        return point_set


    def __len__(self):
        return len(self.samples_paths)
