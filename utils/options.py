from utils.attr_dict import AttrDict
import torch, os
import numpy as np

opts = AttrDict()

opts.model_name = 'STNet'
opts.which_dataset = ['KITTI', 'NUSCENES', 'WAYMO'][0]
opts.train_test = ['train', 'test'][0]
opts.use_tiny = [True, False][1]
opts.reference_BB = ['previous_result', 'previous_gt', 'ground_truth'][0]

opts.device = torch.device("cuda")
opts.batch_size = 48
opts.n_workers = 12
opts.n_epoches = 60
opts.n_gpus = 2
opts.learning_rate = 0.001
opts.subsample_number = 1024
opts.min_points_num = 20
opts.IoU_Space = 3
opts.seed = 1
opts.is_completion = True

# backbone
opts.n_input_feats = 0
opts.use_xyz = True

# attention
opts.feat_emb = 32
opts.iters = 2
opts.knn_num = 48

# voxel
opts.offset_BB = np.array([2, 2, 1])
opts.scale_BB = np.array([1, 1, 1])
opts.voxel_size = [0.3, 0.3, 0.3]
opts.xy_size = [0.3, 0.3]
opts.area_extents = [-5.6, 5.6, -3.6, 3.6, -2.4, 2.4]
opts.xy_area_extents = [-5.6, 5.6, -3.6, 3.6]
opts.downsample = 1.0
opts.regress_radius = 2

opts.ncols = 150

## dataset
opts.db = AttrDict(
    KITTI = AttrDict(
        data_dir = "/opt/data/common/kitti_tracking/kitti_t_o/training/",
        val_data_dir = "/opt/data/common/kitti_tracking/kitti_t_o/training/",
        category_name = ["Car", "Pedestrian", "Van", "Cyclist"][0],
    ),
    NUSCENES = AttrDict(
        data_dir = "/opt/data/common/nuScenes/KITTI_style/train_track",
        val_data_dir = "/opt/data/common/nuScenes/KITTI_style/val",
        category_name = ["car", "pedestrian", "truck", "bicycle"][0],
    ),
    WAYMO = AttrDict(
        data_dir = "/opt/data/common/waymo/sot/",
        val_data_dir = "/opt/data/common/waymo/sot/",
        category_name = ["vehicle", "pedestrian", "cyclist"][0],
    )
)
