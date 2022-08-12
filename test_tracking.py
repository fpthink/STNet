import os
import json
import torch
import numpy as np

from utils.metrics import AverageMeter
from utils.show_line import print_info
from datasets.get_stnet_db import get_dataset
from modules.stnet import STNet_Tracking
from trainers.tester import test_model_kitti_format, test_model_waymo_format

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def test_tracking(opts):
    # nuscenes have the same data format as kitti
    # in the future, we expect to unify the data format between waymo and kitti/nuscenes
    if opts.which_dataset.upper() == 'WAYMO':
        test_tracking_waymo_format(opts)
    else:
        test_tracking_kitti_format(opts)  
    
def test_tracking_kitti_format(opts):  
    ## Init
    print_info(opts.ncols, 'Start')
    set_seed(opts.seed)
    
    ## Define dataset
    print_info(opts.ncols, 'Define datasets')
    test_loader, test_db = get_dataset(opts, partition="Test", shuffle=False)
    opts.voxel_size = torch.from_numpy(test_db.voxel_size.copy()).float()
    opts.voxel_area = test_db.voxel_grid_size
    opts.scene_ground = torch.from_numpy(test_db.scene_ground.copy()).float()
    opts.min_img_coord = torch.from_numpy(test_db.min_img_coord.copy()).float()
    opts.xy_size = torch.from_numpy(test_db.xy_size.copy()).float()

    ## Define model
    print_info(opts.ncols, 'Load model: %s'%opts.model_path)
    model = STNet_Tracking(opts)
    
    if opts.model_path != '':
        try:
            model.load_state_dict(torch.load(opts.model_path))
        except:
            state_dict_ = torch.load(opts.model_path, map_location=lambda storage, loc: storage)
            state_dict = {}
            for k in state_dict_:
                if k.startswith('module') and not k.startswith('module_list'):
                    state_dict[k[7:]] = state_dict_[k]
                else:
                    state_dict[k] = state_dict_[k]
            model.load_state_dict(state_dict, strict=True)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
        
    model.cuda()

    ## online tracking test
    total_lenth = test_db.get_tracklet_framse()
    
    print_info(opts.ncols, 'Start tracking!')
    interval = {'car': [0, 150, 1000, 2500], 'pedestrian': [0, 100, 500, 1000], 'van': [0, 150, 1000, 2500], 'cyclist': [0, 100, 500, 1000]}
    interval_nuscenes = {'car': [0, 150, 1000, 2500], 'pedestrian': [0, 100, 500, 1000], 'truck': [0, 150, 1000, 2500], 'bicycle': [0, 100, 500, 1000]}
    if opts.which_dataset.upper()=='NUSCENES':
        opts.sparse_interval=interval_nuscenes[opts.db.category_name.lower()]
    else:
        opts.sparse_interval=interval[opts.db.category_name.lower()]

    Succ, Prec = test_model_kitti_format(opts, model, test_loader, total_lenth)

    print_info(opts.ncols, 'Total Succ/Prec: %.2f/%.2f'%(Succ, Prec))
        
def test_tracking_waymo_format(opts):  
    ## Init
    print_info(opts.ncols, 'Init voxel opts')
    set_seed(opts.seed)
    init_voxel_opts(opts)
    
    ## Define model
    print_info(opts.ncols, 'Load model: %s'%opts.model_path)
    model = STNet_Tracking(opts)
    if opts.model_path != '':
        try:
            model.load_state_dict(torch.load(opts.model_path))
        except:
            state_dict_ = torch.load(opts.model_path, map_location=lambda storage, loc: storage)
            state_dict = {}
            for k in state_dict_:
                if k.startswith('module') and not k.startswith('module_list'):
                    state_dict[k[7:]] = state_dict_[k]
                else:
                    state_dict[k] = state_dict_[k]
            model.load_state_dict(state_dict, strict=True)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.cuda()

    ## Test tracking
    print_info(opts.ncols, 'Start tracking!')
    data_folder = opts.db.data_dir
    
    bench_paths = []
    bench_lists = []
    json_names = ['easy.json', 'medium.json', 'hard.json', 'bench_list.json']
    
    for json_name in json_names:
        bench_paths.append(os.path.join(data_folder, 'benchmark/validation/', opts.db.category_name,  json_name))
    for bench_path in bench_paths:
        bench_lists.append(json.load(open(bench_path, 'r')))
    
    easy_id_list = []
    medium_id_list = []
    hard_id_list = []
    for tracklet_info in bench_lists[0]:
        easy_id_list.append(tracklet_info['id'])
    for tracklet_info in bench_lists[1]:
        medium_id_list.append(tracklet_info['id'])
    for tracklet_info in bench_lists[2]:
        hard_id_list.append(tracklet_info['id'])
    
    Success_run = AverageMeter()
    Precision_run = AverageMeter()
    Success_easy_run = AverageMeter()
    Precision_easy_run = AverageMeter()
    Success_medium_run = AverageMeter()
    Precision_medium_run = AverageMeter()
    Success_hard_run = AverageMeter()
    Precision_hard_run = AverageMeter()
    
    easy_frame_num = 0
    medium_frame_num = 0
    hard_frame_num = 0
    total_frame_num = 0
    
    passed_num = 0
    
    for tracklet_index, tracklet_info in enumerate(bench_lists[-1]):
        opts.db.tracklet_id = tracklet_info['id']
        opts.db.segment_name = tracklet_info['segment_name']
        opts.db.frame_range = tracklet_info['frame_range']
        
        _, wod_dataset = get_dataset(opts, partition="Test")
        tracklet_length = wod_dataset.get_tracklet_lenth()
        
        print('Prog:({:4d}/{:4d}), ID:"{:}", Len:{:3d}, '\
            .format(tracklet_index + 1, len(bench_lists[-1]), tracklet_info['id'], tracklet_length), end='')

        '''
        There are bus/truck and other instance in the vehicle category of waymo
        and their length may even exceed 10 meters
        but the car category of kitti will not exceed 5.5 meters at most
        '''
        # box_lenth = wod_dataset.get_instance_lenth()
        # if box_lenth > 7.0:
        #     print('Sorry, this vehicle is too long ({:.2f} m)! Pass.'.format(box_lenth))
        #     passed_num += 1
        #     continue
            
        Succ, Prec = test_model_waymo_format(opts=opts, model=model, dataset=wod_dataset)
        
        Success_run.update(Succ, n=tracklet_length)
        Precision_run.update(Prec, n=tracklet_length)
        
        total_frame_num += tracklet_length
        if opts.db.tracklet_id in easy_id_list:
            easy_frame_num += tracklet_length
            Success_easy_run.update(Succ, n=tracklet_length)
            Precision_easy_run.update(Prec, n=tracklet_length)
        elif opts.db.tracklet_id in medium_id_list:
            medium_frame_num += tracklet_length
            Success_medium_run.update(Succ, n=tracklet_length)
            Precision_medium_run.update(Prec, n=tracklet_length)
        else:
            hard_frame_num += tracklet_length
            Success_hard_run.update(Succ, n=tracklet_length)
            Precision_hard_run.update(Prec, n=tracklet_length)
        
        # T_F ==> total frames 
        # C_S/P, T_S/P ==> current success/precision, total success/precision
        print('T_F: %6d (%5d, %5d, %5d), '%(total_frame_num, easy_frame_num, medium_frame_num, hard_frame_num), end='')
        print('C_S/P %5.2f/%5.2f, T_S/P %4.1f/%4.1f (%4.1f/%4.1f, %4.1f/%4.1f, %4.1f/%4.1f)'\
            %(Succ, Prec, Success_run.avg, Precision_run.avg, Success_easy_run.avg, Precision_easy_run.avg, \
                Success_medium_run.avg, Precision_medium_run.avg, Success_hard_run.avg, Precision_hard_run.avg, ))

    print('mean Succ/Prec %.2f/%.2f '%(Success_run.avg, Precision_run.avg))
    # print('There are %d object is too long.'%(passed_num))
    
def init_voxel_opts(opts):
    voxel_size = np.array(opts.voxel_size)
    area_extents = np.array(opts.area_extents).reshape(3, 2)
    xy_size = np.array(opts.xy_size) * opts.downsample
    xy_area_extents = np.array(opts.xy_area_extents).reshape(2, 2)
    extents_transpose = np.array(xy_area_extents).transpose()
    if extents_transpose.shape != (2, 2):
        raise ValueError("Extents are the wrong shape {}".format(extents_transpose.shape))
    # Set image grid extents
    min_img_coord = np.floor(extents_transpose[0] / xy_size)
    voxel_extents_transpose = area_extents.transpose()
    scene_ground = voxel_extents_transpose[0]
    voxel_grid_size = np.ceil(voxel_extents_transpose[1] / voxel_size) - np.floor(voxel_extents_transpose[0] / voxel_size)
    voxel_grid_size = voxel_grid_size.astype(np.int32)
    
    opts.voxel_size = torch.from_numpy(voxel_size.copy()).float()
    opts.voxel_area = voxel_grid_size
    opts.scene_ground = torch.from_numpy(scene_ground.copy()).float()
    opts.min_img_coord = torch.from_numpy(min_img_coord.copy()).float()
    opts.xy_size = torch.from_numpy(xy_size.copy()).float()