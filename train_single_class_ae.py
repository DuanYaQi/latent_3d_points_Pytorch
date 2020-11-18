# 导入相关库
import os.path as osp
import torch
from torch.utils.data import TensorDataset,DataLoader
from torchkeras import Model
import torch.nn as nn
import torch.nn.functional as F

from utils.in_out import snc_category_to_synth_id,load_all_point_clouds_under_folder
from utils.dataset import ShapeNetDataset

class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 256, 1)
        self.conv5 = torch.nn.Conv1d(256, 128, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 6144)

    
    def forward(self,x):
        batchsize = x.size()[0]
        pointnum = x.size()[1]
        channel = x.size()[2]
        z = x.transpose(2, 1)
        z = F.relu(self.bn1(self.conv1(z)))
        z = F.relu(self.bn2(self.conv2(z)))
        z = F.relu(self.bn3(self.conv3(z)))
        z = F.relu(self.bn4(self.conv4(z)))
        z = F.relu(self.bn5(self.conv5(z)))
        z = torch.max(z, 2, keepdim=True)[0]
        z = z.view(-1, 128)

        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = self.fc3(z)

        z = z.view(-1, channel, pointnum)
        z = z.transpose(2, 1)
        return z

    def loss_func(self, z, x):  
        return nn.BCELoss()(z, x)

    # 优化器
    @property
    def optimizer(self):
        return torch.optim.Adam(self.parameters(),lr = 0.0005)

## 预配置
top_out_dir = '/home/latent_3d_points_Pytorch/data/'          # Use to save Neural-Net check-points etc 用于保存神经网络检查点等
top_in_dir = '/home/latent_3d_points_Pytorch/data/shape_net_core_uniform_samples_2048/' # Top-dir of where point-clouds are stored.点云的存储位置的top-dir。
experiment_name = 'single_class_ae'
n_pc_points = 2048                # Number of points per model.每个模型的点数。
bneck_size = 128                  # Bottleneck-AE size    Bottlenck-AE的大小
ae_loss = 'chamfer'                   # Loss to optimize: 'emd' or 'chamfer' 优化损失：'emd' or 'chamfer'
class_name = 'chair'.lower()




# Load Point-Clouds 加载点云
syn_id = snc_category_to_synth_id()[class_name]  # 每个class对应一个文件夹id
class_dir = osp.join(top_in_dir , syn_id)        # 组成class的文件id

dataset = ShapeNetDataset(samples_dir = class_dir)

# 导入训练集数据
dataloader = DataLoader(dataset, batch_size = 50, shuffle=False, num_workers=0)


# 训练一次

def train_step(model, features):
    
    # 正向传播求损失
    predictions = model(features)
    loss = model.loss_func(predictions,feature)
    
    # 反向传播求梯度
    loss.backward()
    
    # 更新模型参数
    model.optimizer.step()
    model.optimizer.zero_grad()
    
    return loss.item()

# 测试train_step效果


features = next(iter(dataloader))
model = EncoderDecoder()
train_step(model,features)
