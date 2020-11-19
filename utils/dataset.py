import torch.utils.data as data
import os
import json

class ShapeNetDataset(data.Dataset):
    def __init__(self,samples_dir):
        self.samples_dir = samples_dir
        self.samples_paths = os.listdir(samples_dir)
       

    def __getitem__(self, index):
        fn = self.samples_paths[index]
        point_set = np.loadtxt(fn[1]).astype(np.float32)

        #resample
        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        point_set = torch.from_numpy(point_set)
        return point_set


    def __len__(self):
        return len(self.datapath)