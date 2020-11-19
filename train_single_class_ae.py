# 导入相关库
import os.path as osp
import torch
from torch.utils.data import TensorDataset,DataLoader
from torchkeras import Model
import torch.nn as nn
import torch.nn.functional as F
import os
import datetime
import warnings

from argparse import ArgumentParser

from utils.in_out import snc_category_to_synth_id
from utils.dataset import ShapeNetDataset
from utils.plot_3d_pc import plot_3d_point_cloud
from metric.loss import ChamferLoss


# -----------------------------------------------------------------------------------------
# 打印时间
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)

# --------------------------------------------------------------------------------------AE变分器
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
        loss = ChamferLoss()
        cd = loss(z,x)
        return cd

    # 优化器
    @property
    def optimizer(self):
        return torch.optim.Adam(self.parameters(),lr = 0.0005)

# -----------------------------------------------------------------------------------------
def train_step(model, features):
    
    # 正向传播求损失
    predictions = model(features)
    loss = model.loss_func(predictions,features)
    
    # 反向传播求梯度
    loss.backward()
    
    # 更新模型参数
    model.optimizer.step()
    model.optimizer.zero_grad()
    
    return loss.item()

# -----------------------------------------------------------------------------------------
# 训练模型
def train_model(model, dataloader, epochs):
    for epoch in range(1,epochs+1):
        for features in dataloader:
            loss = train_step(model,features)
        if epoch%1==0:
            printbar()
            print("epoch =",epoch,"loss = ",loss)

# -----------------------------------------------------------------------------------------
def showfig(model, dataloader):
    feed_pc = next(iter(dataloader))
    reconstructions = model(feed_pc)
    reconstructions = reconstructions.detach()

    i = 1
    # Ground Truth
    plot_3d_point_cloud(feed_pc[i][:, 0], 
                        feed_pc[i][:, 1], 
                        feed_pc[i][:, 2], in_u_sphere=True);
    # Generative Point
    plot_3d_point_cloud(reconstructions[i][:, 0], 
                        reconstructions[i][:, 1], 
                        reconstructions[i][:, 2], in_u_sphere=True);

# -----------------------------------------------------------------------------------------
def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--top_in_dir', type=str, help='Top-dir of where point-clouds are stored', default = '/home/latent_3d_points_Pytorch/data/shape_net_core_uniform_samples_2048/')
    parser.add_argument('--n_pc_points', type=int, help='Number of points per model', default = 2048)       #TODO: Adapt datasets
    parser.add_argument('--bneck_size', type=int, help='Bottleneck-AE size', default = 128)                 #TODO: Adapt haparms
    parser.add_argument('--ae_loss', type=str, help='Loss to optimize: emd or chamfer', default = 'chamfer') #TODO: ADD EMD
    parser.add_argument('--class_name', type=str, default = 'chair')
    parser.add_argument('--batch_size', type=int, default = 50)
    parser.add_argument('--sample_num', type=int, default = 100)
    parser.add_argument('--epochs', type=int, default = 1)
    return parser.parse_args()

# -----------------------------------------------------------------------------------------
def train(phase='Train', checkpoint_path: str=None):
    args = parse_arguments()

    # Load Point-Clouds 加载点云
    syn_id = snc_category_to_synth_id()[args.class_name]  # 每个class对应一个文件夹id
    class_dir = osp.join(args.top_in_dir , syn_id)        # 组成class的文件id

    # 导入训练集数据
    dataset = ShapeNetDataset(samples_dir = class_dir, sample_num = args.sample_num) # TODO: set your own sample_num 
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle=False, num_workers=0)
    model = EncoderDecoder()

    if phase == 'Train':
        train_model(model, dataloader, args.epochs)
        
        if checkpoint_path is not None:
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Model has been save to \033[1m{checkpoint_path}\033[0m')

    else:  # Test
        model.load_state_dict(torch.load(checkpoint_path))

    showfig(model, dataloader)
    
    

# -----------------------------------------------------------------------------------------
if __name__ == "__main__":

    checkpoint_path = './model/AEModel1.pkl'
    train('Train', checkpoint_path)
    #train('Test', checkpoint_path)