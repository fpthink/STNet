import os
import time
import torch
import pickle
import numpy as np

from utils.metrics import AverageMeter, Success, Precision
from utils.metrics import estimateOverlap, estimateAccuracy
from utils import kitti_utils
from utils.decode import mot_decode

def test_model_kitti_format(opts, model, dataloder, total_lenth):
    opts.sparse_interval
    # Defining statistical objects
    batch_time = AverageMeter()         # network forward
    data_time = AverageMeter()          # data 
    post_process_time = AverageMeter()  # pose

    Success_main = Success()
    Precision_main = Precision()
    Success_batch = Success()
    Precision_batch = Precision()
    Success_key_main = Success()
    Precision_key_main = Precision()
    Success_key_batch = Success()
    Precision_key_batch = Precision()
    
    cur_lenth = 0

    area_extents = np.array(opts.area_extents).reshape(3, 2)
    
    # Switch to evaluate mode
    dataset = dataloder.dataset
    model.eval()
    
    end = time.time()

    tracklet_nums = 0
    track_anno_nums = 0
    key_track_anno_nums = 0
    
    # with tqdm(enumerate(dataloder), total=len(dataloder.dataset.list_of_anno)) as t:
    for batch_id, batch in enumerate(dataloder):   
        # For each tracklet, do the tracking test
        for PCs, BBs, list_of_anno in batch: 
            results_BBs = []
            if dataset.which_dataset=='NUSCENES':
                if list_of_anno[0]['num_lidar_pts']==0:
                    break
            this_key_track_anno_nums = 0
            track_anno_nums += len(PCs)
            tracklet_nums += 1

            for i, _ in enumerate(PCs):
                this_anno = list_of_anno[i]
                this_BB = BBs[i]
                this_PC = PCs[i]
                if dataset.which_dataset=='NUSCENES' and this_anno['is_key_frame'] == 1:
                    key_track_anno_nums += 1
                    this_key_track_anno_nums += 1
                    
                # step 1. Initial frame
                if i == 0:
                    # the first frame of current tracklet, we can get the grount truth bounding box
                    results_BBs.append(BBs[i])
                else:
                    # others frame, we need use our model to predict the bounding box
                    cur_PC = PCs[i]     
                    
                    # step 2. Get the previoud/reference bounding box
                    if ("previous_result".upper() in opts.reference_BB.upper()):
                        pre_BB = results_BBs[-1]
                    elif ("previous_gt".upper() in opts.reference_BB.upper()):
                        pre_BB = BBs[i-1]  
                    else:
                        pre_BB = BBs[i]
                        
                    # step 3. Get the point cloud
                    target_PC = kitti_utils.cropAndCenterPC(cur_PC, pre_BB, offset=dataset.offset_BB, scale=1.25, limit_area=area_extents)      # (3, N2)
                    model_PC = kitti_utils.getModel([PCs[0], PCs[i-1]], [results_BBs[0], results_BBs[i-1]], scale=1.25)
                    # step 3.1 translate to numpy
                    target_PC = target_PC.points    # (3, N2)
                    model_PC = np.array(model_PC.points, dtype=np.float32)
                    # step 3.2 subsample
                    target_PC = kitti_utils.subsamplePC(target_PC, dataset.subsample_number)    # (M  , 3), tensor
                    model_PC = kitti_utils.subsamplePC(model_PC, dataset.subsample_number//2)   # (M/2, 3), tensor
                    
                    data_time.update(time.time() - end)
                    end = time.time()
                    
                    # step 4. Regression
                    pred_hm, pred_loc, pred_z_axis = model( model_PC.unsqueeze(0).cuda(), 
                                                            target_PC.unsqueeze(0).cuda())  
                    
                    batch_time.update(time.time() - end)
                    end = time.time()

                    # step 5. Get current bounding box
                    with torch.no_grad():
                        hm = pred_hm.sigmoid_()
                        xy_img_z_ry = mot_decode(hm,pred_loc,pred_z_axis,K=1)

                    xy_img_z_ry_cpu = xy_img_z_ry.squeeze(0).detach().cpu().numpy()
                    xy_img_z_ry_cpu[:,:2] = (xy_img_z_ry_cpu[:,:2]+dataset.min_img_coord)*dataset.xy_size
                    estimate_box = xy_img_z_ry_cpu[0]

                    box = kitti_utils.getOffsetBBtest(pre_BB, estimate_box[:4], category=opts.db.category_name)
                    results_BBs.append(box)

                # step 6. Estimate overlap/accuracy for current sample
                # the BBs[i] is gournd truth box, and the results_BBs[-1] is model predict box
                this_overlap = estimateOverlap(BBs[i], results_BBs[-1], dim=opts.IoU_Space)
                this_accuracy = estimateAccuracy(BBs[i], results_BBs[-1], dim=opts.IoU_Space)

                Success_main.add_overlap(this_overlap)
                Precision_main.add_accuracy(this_accuracy)
                Success_batch.add_overlap(this_overlap)
                Precision_batch.add_accuracy(this_accuracy)
                if dataset.which_dataset == 'NUSCENES' and this_anno['is_key_frame']==1:
                    Success_key_main.add_overlap(this_overlap)
                    Precision_key_main.add_accuracy(this_accuracy)
                    Success_key_batch.add_overlap(this_overlap)
                    Precision_key_batch.add_accuracy(this_accuracy)

                # measure elapsed time
                post_process_time.update(time.time() - end)
                end = time.time()

            cur_lenth += len(PCs)
            # batch end
            if dataset.which_dataset == 'NUSCENES':
                print(  'Tracklet ID {:3d}: '.format(batch_id)+
                        'Length:({:3d}/{:3d}), '.format(this_key_track_anno_nums, len(PCs))+
                        'Data:{:4.1f}ms '.format(1000*data_time.avg) +
                        'Forward:{:5.1f}ms '.format(1000*batch_time.avg) +
                        'Pose:{:4.1f}ms '.format(1000*post_process_time.avg) +
                        'Succ/Prec:'+
                        '{:5.1f}/'.format(Success_key_batch.average)+
                        '{:5.1f} '.format(Precision_key_batch.average)+
                        '(Total:{:5.1f}/'.format(Success_key_main.average)+
                        '{:5.1f}), '.format(Precision_key_main.average)+
                        'Key_nums prog:({:4d}/ {:4d}), '.format(key_track_anno_nums, track_anno_nums)
                    )
            else:
                print(  'Tracklet ID {:3d}: '.format(batch_id)+
                        'Length:{:3d}, '.format(len(PCs))+
                        'Data:{:4.1f}ms '.format(1000*data_time.avg) +
                        'Forward:{:5.1f}ms '.format(1000*batch_time.avg) +
                        'Pose:{:4.1f}ms '.format(1000*post_process_time.avg) +
                        'Succ/Prec:'+
                        '{:5.1f}/'.format(Success_batch.average)+
                        '{:5.1f} '.format(Precision_batch.average)+
                        '(Total:{:5.1f}/'.format(Success_main.average)+
                        '{:5.1f}), '.format(Precision_main.average)+
                        'Progress:{:6.2f}%'.format(100*cur_lenth/total_lenth)
                    )
                
            # batch reset
            Success_batch.reset()
            Precision_batch.reset()
            Success_key_batch.reset()
            Precision_key_batch.reset()
    if dataset.which_dataset == 'NUSCENES':
        return Success_key_main.average, Precision_key_main.average
    return Success_main.average, Precision_main.average
    
def test_model_waymo_format(opts, model, dataset):
    reference_BB = opts.reference_BB
    IoU_Space = opts.IoU_Space
    subsample_number = opts.subsample_number
    offset_BB = opts.offset_BB
    
    area_extents = opts.area_extents
    xy_size = opts.xy_size.numpy()
    min_img_coord = opts.min_img_coord.numpy()   

    Success_main = Success()
    Precision_main = Precision()

    area_extents = np.array(area_extents).reshape(3, 2)
    
    # Switch to evaluate mode
    model.eval()
    
    PCs, BBs = dataset.get_PCs_and_BBs()
    results_BBs = []
    
    for i in range(len(PCs)):
        if i == 0:
            # the first frame of current tracklet, we can get the grount truth bounding box
            results_BBs.append(BBs[i])
        else:
            # others frame, we need use our model to predict the bounding box
            cur_PC = PCs[i]     # current  frame's point cloud, (3, n2)
            
            # step 2. Get the previoud/reference bounding box
            if ("previous_result".upper() in reference_BB.upper()):
                pre_BB = results_BBs[-1]
            elif ("previous_gt".upper() in reference_BB.upper()):
                pre_BB = BBs[i-1]  
            else:
                pre_BB = BBs[i]
                
            # step 3. Get the point cloud
            target_PC = kitti_utils.cropAndCenterPC(cur_PC, pre_BB, offset=offset_BB, scale=1.25, limit_area=area_extents)      # (3, N2)
            model_PC = kitti_utils.getModel([PCs[0], PCs[i-1]], [results_BBs[0], results_BBs[i-1]], scale=1.25)
            # step 3.1 translate to numpy
            target_PC = target_PC.points    # (3, N2)
            model_PC = np.array(model_PC.points, dtype=np.float32)
            # step 3.2 subsample
            target_PC = kitti_utils.subsamplePC(target_PC, subsample_number)    # (M  , 3), tensor
            model_PC = kitti_utils.subsamplePC(model_PC, subsample_number//2)   # (M/2, 3), tensor
            
            # step 4. Regression
            pred_hm, pred_loc, pred_z_axis = model(model_PC.unsqueeze(0).cuda(), 
                                                    target_PC.unsqueeze(0).cuda())  

            # step 5. Get current bounding box
            with torch.no_grad():
                hm = pred_hm.sigmoid_()
                xy_img_z_ry = mot_decode(hm,pred_loc,pred_z_axis,K=1)

            xy_img_z_ry_cpu = xy_img_z_ry.squeeze(0).detach().cpu().numpy()
            xy_img_z_ry_cpu[:,:2] = (xy_img_z_ry_cpu[:,:2]+min_img_coord)*xy_size
            estimate_box = xy_img_z_ry_cpu[0]

            box = kitti_utils.getOffsetBBtest(pre_BB, estimate_box[:4])
            results_BBs.append(box)

        # step 6. Estimate overlap/accuracy fro current sample
        # the BBs[i] is gournd truth box, and the results_BBs[-1] is model predict box
        this_overlap = estimateOverlap(BBs[i], results_BBs[-1], dim=IoU_Space, dataset_type='waymo')
        this_accuracy = estimateAccuracy(BBs[i], results_BBs[-1], dim=IoU_Space)
        Success_main.add_overlap(this_overlap)
        Precision_main.add_accuracy(this_accuracy)

    return Success_main.average, Precision_main.average