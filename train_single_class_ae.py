# This py file will help you train a vanilla Point-Cloud AE with the basic architecture we used in Latent_3d_point.
#     (it assumes latent_3d_points is in the PYTHONPATH and the structural losses have been compiled)

# 导入相关库
import os.path as osp
import sys
sys.path.append(osp.abspath('/home/latent_3d_point_Pytorch'))
from latent_3d_points.src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder

from latent_3d_points.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet,                                         load_all_point_clouds_under_folder

from latent_3d_points.src.tf_utils import reset_tf_graph
from latent_3d_points.src.general_utils import plot_3d_point_cloud



# Define Basic Parameters 定义基本参数
top_out_dir = '/home/latent_3d_points/data/'          # Use to save Neural-Net check-points etc 用于保存神经网络检查点等
top_in_dir = '/home/latent_3d_points/data/shape_net_core_uniform_samples_2048/' # Top-dir of where point-clouds are stored.点云的存储位置的top-dir。

experiment_name = 'single_class_ae'
n_pc_points = 2048                # Number of points per model.每个模型的点数。
bneck_size = 128                  # Bottleneck-AE size    Bottlenck-AE的大小
ae_loss = 'chamfer'                   # Loss to optimize: 'emd' or 'chamfer' 优化损失：'emd' or 'chamfer'
class_name = raw_input('Give me the class name (e.g. "chair"): ').lower()


# Load Point-Clouds 加载点云
syn_id = snc_category_to_synth_id()[class_name]  # 每个class对应一个文件夹id
class_dir = osp.join(top_in_dir , syn_id)        # 组成class的文件id
all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True) # 加载文件夹下的全部点云数据


# Load default training parameters (some of which are listed beloq). For more details please print the configuration object.
# 
#     'batch_size': 50   
#     
#     'denoising': False     (# by default AE is not denoising)
# 
#     'learning_rate': 0.0005
# 
#     'z_rotate': False      (# randomly rotate models of each batch)
#     
#     'loss_display_step': 1 (# display loss at end of these many epochs)
#     'saver_step': 10       (# over how many epochs to save neural-network)

train_params = default_train_params()


encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)
train_dir = create_dir(osp.join(top_out_dir, experiment_name))


conf = Conf(n_input = [n_pc_points, 3],
            loss = ae_loss,
            training_epochs = train_params['training_epochs'],
            batch_size = train_params['batch_size'],
            denoising = train_params['denoising'],
            learning_rate = train_params['learning_rate'],
            train_dir = train_dir,
            loss_display_step = train_params['loss_display_step'],
            saver_step = train_params['saver_step'],
            z_rotate = train_params['z_rotate'],
            encoder = encoder,
            decoder = decoder,
            encoder_args = enc_args,
            decoder_args = dec_args
           )
conf.experiment_name = experiment_name
conf.held_out_step = 5   # How often to evaluate/print out loss on 
                         # held_out data (if they are provided in ae.train() ).
conf.save(osp.join(train_dir, 'configuration'))


# If you ran the above lines, you can reload a saved model like this:

load_pre_trained_ae = False
restore_epoch = 500
if load_pre_trained_ae:
    conf = Conf.load(train_dir + '/configuration')
    reset_tf_graph()
    ae = PointNetAutoEncoder(conf.experiment_name, conf)
    ae.restore_model(conf.train_dir, epoch=restore_epoch)


# Build AE Model.

reset_tf_graph()
ae = PointNetAutoEncoder(conf.experiment_name, conf)


# Train the AE (save output to train_stats.txt) 

buf_size = 1 # Make 'training_stats' file to flush each output line regarding training.
fout = open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size)
train_stats = ae.train(all_pc_data, conf, log_file=fout)
fout.close()


# Get a batch of reconstuctions and their latent-codes.

feed_pc, feed_model_names, _ = all_pc_data.next_batch(10)
reconstructions = ae.reconstruct(feed_pc)[0]
latent_codes = ae.transform(feed_pc)


# Use any plotting mechanism such as matplotlib to visualize the results.

i = 2
plot_3d_point_cloud(reconstructions[i][:, 0], 
                    reconstructions[i][:, 1], 
                    reconstructions[i][:, 2], in_u_sphere=True);



i = 4
plot_3d_point_cloud(reconstructions[i][:, 0], 
                    reconstructions[i][:, 1], 
                    reconstructions[i][:, 2], in_u_sphere=True);


