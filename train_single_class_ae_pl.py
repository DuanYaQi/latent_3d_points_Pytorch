import os
import os.path as osp
import time
import datetime
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.utils.data import TensorDataset,DataLoader
import pytorch_lightning as pl

from utils.in_out import snc_category_to_synth_id
from utils.dataset import ShapeNetDataset
from utils.plot_3d_pc import plot_3d_point_cloud
from metric.loss import ChamferLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------------------AE
class AE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.network = EncoderDecoder()

    def forward(self, x):
        z = self.network(x)
        return z

    def training_step(self, batch, batch_idx):
        x = batch
        z = self.network(x)
        loss = self.loss_func(z, x)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def loss_func(self, z, x):  
        loss = ChamferLoss()
        cd = loss(z,x)
        return cd

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 0.0005)
        return optimizer
# --------------------------------------------------------------------------------------
class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, 256, 1)
        self.conv5 = nn.Conv1d(256, 128, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 6144)
    
    def forward(self, x):
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

# -----------------------------------------------------------------------------------------
def showfig(model, dataloader):
    feed_pc = next(iter(dataloader))
    reconstructions = model(feed_pc.to(device))
    if torch.cuda.is_available():
        reconstructions = reconstructions.detach().to("cpu")
    else:
        reconstructions = reconstructions.detach()
    
    i = 16
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
    parser.add_argument('--top_in_dir', type=str, help='Top-dir of where point-clouds are stored', default = '../data/latent_3d_points_Pytorch/shape_net_core_uniform_samples_2048/')
    parser.add_argument('--n_pc_points', type=int, help='Number of points per model', default = 2048)       #TODO: Adapt datasets
    parser.add_argument('--bneck_size', type=int, help='Bottleneck-AE size', default = 128)                 #TODO: Adapt haparms
    parser.add_argument('--ae_loss', type=str, help='Loss to optimize: emd or chamfer', default = 'chamfer') #TODO: ADD EMD
    parser.add_argument('--class_name', type=str, default = 'chair')
    parser.add_argument('--batch_size', type=int, default = 32)
    parser.add_argument('--sample_num', type=int, default = 6400)
    parser.add_argument('--max_epochs', type=int, default = 10)
    return parser.parse_args()

# -----------------------------------------------------------------------------------------
def train(phase='Train', checkpoint_path: str=None):
    args = parse_arguments()
    trainer_config = {
        'gpus'                   : 1,  # Set this to None for CPU training
        'max_epochs'             : args.max_epochs,
        #'automatic_optimization' : True,
    }
    # Load Point-Clouds
    syn_id = snc_category_to_synth_id()[args.class_name]  # class2id
    class_dir = osp.join(args.top_in_dir , syn_id)
    # dataset
    dataset = ShapeNetDataset(samples_dir = class_dir, sample_num = args.sample_num)
    train_loader = DataLoader(dataset, batch_size = args.batch_size, shuffle=False, num_workers=2)
    # network
    autoencoder = AE()
    trainer = pl.Trainer(**trainer_config)

    if phase == 'Train':
        trainer.fit(autoencoder, train_loader)
        if checkpoint_path is not None:
            torch.save(autoencoder.network, checkpoint_path)
            print(f'Model has been save to \033[1m{checkpoint_path}\033[0m')
    elif phase == 'continueTrain':
        autoencoder.network = torch.load(checkpoint_path, map_location=device)
        trainer.fit(autoencoder, train_loader)
        if checkpoint_path is not None:
            torch.save(autoencoder.network, checkpoint_path)
            print(f'Model has been save to \033[1m{checkpoint_path}\033[0m')
    else:
        autoencoder.network = torch.load(checkpoint_path, map_location=device)
        
    showfig(autoencoder, train_loader)

# -----------------------------------------------------------------------------------------
if __name__ == "__main__":
    checkpoint_path = './model/AEModel10epoch.pkl'
    train('Train', checkpoint_path)
    # checkpoint_path = './model/AEModel100epoch.pkl'
    # train('Test', checkpoint_path)
