# 导入相关库
import os.path as osp
from utils.in_out import snc_category_to_synth_id,load_all_point_clouds_under_folder

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
pc_clouds = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True) # 加载文件夹下的全部点云数据


