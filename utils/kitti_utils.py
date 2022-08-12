import pdb
import copy
import torch

import numpy as np
from pyquaternion import Quaternion
from utils.data_classes import PointCloud, BoundingBox

def getlabelPC(PC, box, offset=0, scale=1.0):
    box_tmp = copy.deepcopy(box)
    new_PC = PointCloud(PC.points.copy())

    rot_mat = np.transpose(box_tmp.rotation_matrix)
    trans = -box_tmp.center

    # align data
    new_PC.translate(trans)
    box_tmp.translate(trans)
    new_PC.rotate((rot_mat))
    box_tmp.rotate(Quaternion(matrix=(rot_mat)))
    
    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    x_filt_max = new_PC.points[0, :] < maxi[0]
    x_filt_min = new_PC.points[0, :] > mini[0]
    y_filt_max = new_PC.points[1, :] < maxi[1]
    y_filt_min = new_PC.points[1, :] > mini[1]
    z_filt_max = new_PC.points[2, :] < maxi[2]
    z_filt_min = new_PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    new_label = np.zeros(new_PC.points.shape[1])
    new_label[close] = 1

    return new_label

def cropPC(PC, box, offset, scale, limit_area=None):
    boxTemp = copy.deepcopy(box)
    boxTemp.wlh = boxTemp.wlh * scale   
    
    maxi = np.max(boxTemp.corners(), 1) + offset
    mini = np.min(boxTemp.corners(), 1) - offset

    x_filt_max = PC.points[0, :] < maxi[0]
    x_filt_min = PC.points[0, :] > mini[0]
    y_filt_max = PC.points[1, :] < maxi[1]
    y_filt_min = PC.points[1, :] > mini[1]
    z_filt_max = PC.points[2, :] < maxi[2]
    z_filt_min = PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    if limit_area is not None :
        max_area=np.max(limit_area,1)
        min_area=np.min(limit_area,1)
        x_area_max=PC.points[0, :] < max_area[0]
        x_area_min=PC.points[0, :] > min_area[0]
        y_area_max=PC.points[1, :] < max_area[1]
        y_area_min=PC.points[1, :] > min_area[1]
        z_area_max=PC.points[2, :] < max_area[2]
        z_area_min=PC.points[2, :] > min_area[2]

        close = np.logical_and(close, x_area_min)
        close = np.logical_and(close, x_area_max)
        close = np.logical_and(close, y_area_min)
        close = np.logical_and(close, y_area_max)
        close = np.logical_and(close, z_area_min)
        close = np.logical_and(close, z_area_max)

    afterCropPC = PointCloud(PC.points[:, close])
    return afterCropPC

def cropPCwithlabel(PC, box, label, offset=0, scale=1.0, limit_area=None):
    box_tmp = copy.deepcopy(box)
    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    x_filt_max = PC.points[0, :] < maxi[0]
    x_filt_min = PC.points[0, :] > mini[0]
    y_filt_max = PC.points[1, :] < maxi[1]
    y_filt_min = PC.points[1, :] > mini[1]
    z_filt_max = PC.points[2, :] < maxi[2]
    z_filt_min = PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    if limit_area is not None :
        max_area=np.max(limit_area,1)
        min_area=np.min(limit_area,1)
        x_area_max=PC.points[0, :] < max_area[0]
        x_area_min=PC.points[0, :] > min_area[0]
        y_area_max=PC.points[1, :] < max_area[1]
        y_area_min=PC.points[1, :] > min_area[1]
        z_area_max=PC.points[2, :] < max_area[2]
        z_area_min=PC.points[2, :] > min_area[2]

        close = np.logical_and(close, x_area_min)
        close = np.logical_and(close, x_area_max)
        close = np.logical_and(close, y_area_min)
        close = np.logical_and(close, y_area_max)
        close = np.logical_and(close, z_area_min)
        close = np.logical_and(close, z_area_max)

    new_PC = PointCloud(PC.points[:, close])
    new_label = label[close]

    return new_PC, new_label

def cropAndCenterPC(PC, box, offset, scale, normalize=False, limit_area=None):
    new_PC = cropPC(PC, box, offset=2 * offset, scale=4 * scale)
    new_box = copy.deepcopy(box)

    rot_mat = np.transpose(new_box.rotation_matrix)
    
    # align data
    trans = -new_box.center
    new_PC.translate(trans)
    new_box.translate(trans)

    new_PC.rotate((rot_mat))
    new_box.rotate(Quaternion(matrix=(rot_mat)))

    # crop around box
    new_PC = cropPC(new_PC, new_box, offset=offset, scale=scale, limit_area=limit_area)

    if normalize:
        new_PC.normalize(box.wlh)
    return new_PC

def cropAndCenterPC_withClaAndOff(PC, cut_box, gt_box, offset, scale, limit_area=None):
    new_PC = cropPC(PC, cut_box, offset=2 * offset, scale=4 * scale)

    new_box_cut = copy.deepcopy(cut_box)
    new_box_gt = copy.deepcopy(gt_box)

    class_label = getlabelPC(new_PC, gt_box, scale=np.array([1.10, 1.05, 1.025]))

    rot_mat = np.transpose(new_box_cut.rotation_matrix)
    trans = -new_box_cut.center
    
    # align data
    new_PC.translate(trans)
    new_box_cut.translate(trans)
    new_box_gt.translate(trans)
    new_PC.rotate((rot_mat))
    new_box_cut.rotate(Quaternion(matrix=(rot_mat)))
    new_box_gt.rotate(Quaternion(matrix=(rot_mat)))

    # crop around box
    new_PC, class_label = cropPCwithlabel(new_PC, new_box_cut, class_label, offset=offset, scale=scale, limit_area=limit_area)

    # get offset(center & rotation)
    offcenter = np.array([new_box_gt.center[0], new_box_gt.center[1], new_box_gt.center[2], new_box_gt.orientation.degrees*new_box_gt.orientation.axis[2]])

    return new_PC, class_label, offcenter, new_box_gt

def getOffsetBBtest(box, offset, category='Car'):
    rot_quat = Quaternion(matrix=box.rotation_matrix)
    trans = np.array(box.center)

    new_box = copy.deepcopy(box)

    new_box.translate(-trans)
    new_box.rotate(rot_quat.inverse)

    # REMOVE TRANSfORM
    if len(offset) == 3:
        new_box.rotate(
            Quaternion(axis=[0, 0, 1], angle=offset[2] * np.pi / 180))
    elif len(offset) == 4:
        new_box.rotate(
            Quaternion(axis=[0, 0, 1], angle=offset[3] * np.pi / 180))
        
    if category.lower() == 'cyclist':
        if offset[0]>min(new_box.wlh[1],2):
            offset[0] = np.random.uniform(-1,1)
        if offset[1]>min(new_box.wlh[0],2):
            offset[1] = np.random.uniform(-1,1)
    else:
        if offset[0]>new_box.wlh[0]:
            offset[0] = np.random.uniform(-1,1)
        if offset[1]>min(new_box.wlh[1],2):
            offset[1] = np.random.uniform(-1,1)
        
    if len(offset) == 4:
        if offset[2]>min(new_box.wlh[2],0.5):
            offset[2] = np.random.uniform(-0.5,0.5)
        new_box.translate(np.array([offset[0], offset[1], offset[2]]))
    else:
        new_box.translate(np.array([offset[0], offset[1], 0]))
    # APPLY PREVIOUS TRANSFORMATION
    new_box.rotate(rot_quat)
    new_box.translate(trans)
    return new_box

def getOffsetBB_data(box, offset):
    rot_quat = Quaternion(matrix=box.rotation_matrix)
    trans = np.array(box.center)

    new_box = copy.deepcopy(box)

    new_box.translate(-trans)
    new_box.rotate(rot_quat.inverse)

    # REMOVE TRANSfORM
    if len(offset) == 3:
        new_box.rotate(
            Quaternion(axis=[0, 0, 1], angle=offset[2] * np.pi / 180))
    elif len(offset) == 4:
        new_box.rotate(
            Quaternion(axis=[0, 0, 1], angle=offset[3] * np.pi / 180))
    if np.abs(offset[0])>min(new_box.wlh[1],2):
        offset[0] = np.random.uniform(-1,1)
    if np.abs(offset[1])>min(new_box.wlh[0],2):
        offset[1] = np.random.uniform(-1,1)
    if len(offset) == 4:
        if np.abs(offset[2])>min(new_box.wlh[2],0.5):
            offset[2] = np.random.uniform(-0.5,0.5)
        new_box.translate(np.array([offset[0], offset[1], offset[2]]))
    else:
        new_box.translate(np.array([offset[0], offset[1], 0]))
    # APPLY PREVIOUS TRANSFORMATION
    new_box.rotate(rot_quat)
    new_box.translate(trans)
    return new_box

def box_local(PC, box, offset, scale):
    new_PC = PointCloud(PC)
    new_box = copy.deepcopy(box)

    # crop around box
    class_label = getlabelPC(new_PC, new_box, offset, scale)

    local_idxs = np.arange(class_label.shape[0])[class_label==1]

    surf_class_label = getlabelPC(new_PC, new_box, scale=np.array([1.10, 1.05, 1.025]))

    return torch.from_numpy(surf_class_label), torch.from_numpy(local_idxs)

def get_PC_offset(cur_BB, pre_BB):
    new_cur_box = copy.deepcopy(cur_BB)
    new_pre_box = copy.deepcopy(pre_BB)

    rot_mat = np.transpose(new_pre_box.rotation_matrix)
    trans = -new_pre_box.center

    # rotate and translate to pre_BB center
    new_pre_box.translate(trans)     
    new_pre_box.rotate(Quaternion(matrix=(rot_mat)))
    new_cur_box.translate(trans)
    new_cur_box.rotate(Quaternion(matrix=(rot_mat)))

    # center offset
    center_offset = new_cur_box.center
    # rotate radians offset
    degrees_offset = new_cur_box.orientation.rotation_matrix

    rot_mat = np.hstack((degrees_offset, center_offset.reshape(3,1)))   #3*4
    transf_mat = np.vstack((rot_mat, np.array([0, 0, 0, 1])))
    
    return torch.from_numpy(degrees_offset).float(), torch.from_numpy(center_offset).float()

def get_sample_PC_offset(cur_BB, pre_BB, sample_pre_BB):
    new_cur_box = copy.deepcopy(cur_BB)
    new_pre_box = copy.deepcopy(pre_BB)
    new_sample_pre_box = copy.deepcopy(sample_pre_BB)

    rot_mat = np.transpose(new_sample_pre_box.rotation_matrix)
    trans = -new_sample_pre_box.center

    # rotate and translate to pre_BB center
    new_pre_box.translate(trans)     
    new_pre_box.rotate(Quaternion(matrix=(rot_mat)))
    new_cur_box.translate(trans)
    new_cur_box.rotate(Quaternion(matrix=(rot_mat)))
    new_sample_pre_box.translate(trans)
    new_sample_pre_box.rotate(Quaternion(matrix=(rot_mat)))

    # center offset
    center_offset = new_cur_box.center - new_pre_box.center
    # rotate radians offset
    pre_BB_rotmat = np.transpose(new_pre_box.rotation_matrix)
    degrees_offset = Quaternion(matrix=np.dot(new_cur_box.rotation_matrix, pre_BB_rotmat)).rotation_matrix
    
    return torch.from_numpy(degrees_offset).float(), torch.from_numpy(center_offset).float()

def subsamplePC(PC, subsample_number):

    if subsample_number == 0:
        pass
    elif PC.shape[1] > 2:
        if PC.shape[0] > 3:
            PC = PC[0:3, :]
        if PC.shape[1] != subsample_number:
            # subsample
            new_pts_idx = np.random.randint(low=0, high=PC.shape[1], size=subsample_number, dtype=np.int64)
            PC = PC[:, new_pts_idx]
        PC = PC.reshape(3, subsample_number)
    else:
        PC = np.zeros((3, subsample_number))
    
    return torch.from_numpy(PC).float().t() #(N, 3)

def subsamplePC_withClaAndOff(PC, class_label, offcenter, subsample_number, cut=False, istrain=True):
    if np.shape(PC)[1] > 2:
        if PC.shape[0] > 3:
            PC = PC[0:3, :]
        if PC.shape[1] != subsample_number:
            if not istrain:
                np.random.seed(1)
            new_pts_idx = np.random.randint(low=0, high=PC.shape[1], size=subsample_number, dtype=np.int64)   # random sample
            PC = PC[:, new_pts_idx]
            class_label = class_label[new_pts_idx]
        PC = PC.reshape((3, subsample_number))
        offcenter = np.tile(offcenter,[np.size(class_label),1])   
    else:
        PC = np.zeros((3, subsample_number))
        class_label = np.zeros(subsample_number)
        offcenter = np.tile(offcenter,[np.size(class_label),1])

    if cut:
        cut_num = PC.shape[-1]//8
        class_label = class_label[:cut_num]
        offcenter = offcenter[:cut_num]

    return torch.from_numpy(PC).float().t(), torch.from_numpy(class_label).float(), torch.from_numpy(offcenter).float()

def getModel(PCs, boxes, offset=0, scale=1.0, normalize=False):
    
    if len(PCs) == 0:
        return PointCloud(np.ones((3, 0)))
    points = np.ones((PCs[0].points.shape[0], 0))

    for PC, box in zip(PCs, boxes):
        cropped_PC = cropAndCenterPC(
            PC, box, offset=offset, scale=scale, normalize=normalize)
        # try:
        if cropped_PC.points.shape[1] > 0:
            points = np.concatenate([points, cropped_PC.points], axis=1)

    PC = PointCloud(points)

    return PC

