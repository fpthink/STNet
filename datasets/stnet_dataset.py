import os

import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion

import torch
from utils import kitti_utils, searchspace, metrics
from utils.data_classes import PointCloud, BoundingBox

from datasets.base_dataset import BaseDataset

class TrainDataset(BaseDataset):
    def __init__(self, opts, split):
        super(TrainDataset, self).__init__(
            which_dataset = opts.which_dataset,
            path = opts.db.data_dir,
            split = split,
            category_name = opts.db.category_name,
            offset_BB = opts.offset_BB,
            scale_BB = opts.scale_BB
        )
        # Number of candidate point clouds per frame
        self.num_candidates_perframe = 4
        
        # Init
        self.offset_BB = opts.offset_BB
        self.scale_BB = opts.scale_BB
        self.templates_num = opts.templates_num
        self.subsample_number = opts.subsample_number
        self.min_points_num = opts.min_points_num
        self.preload_data_path = opts.data_save_path + '/' + self.category_name + '_' + split.split('_')[1]
        
        # voxelize point cloud parameter
        self.voxel_size = np.array(opts.voxel_size)                              
        self.xy_size = np.array(opts.xy_size) * opts.downsample               
        self.area_extents = np.array(opts.area_extents).reshape(3,2)             
        self.xy_area_extents = np.array(opts.xy_area_extents).reshape(2,2)       
        self.regress_radius = opts.regress_radius                                
        self.voxel_extents_transpose = self.area_extents.copy().transpose() 
        self.scene_ground = self.voxel_extents_transpose[0]                 
        self.voxel_grid_size = np.ceil(self.voxel_extents_transpose[1] / self.voxel_size) - np.floor(self.voxel_extents_transpose[0] / self.voxel_size)   
        self.voxel_grid_size = self.voxel_grid_size.astype(np.int32)        

        extents_transpose = self.xy_area_extents.transpose()    
        if extents_transpose.shape != (2, 2):
            raise ValueError("Extents are the wrong shape {}".format(extents_transpose.shape))

        self.min_img_coord = np.floor(extents_transpose[0] / self.xy_size)              
        self.max_img_coord = np.ceil(extents_transpose[1] / self.xy_size) - 1           
        self.img_size = ((self.max_img_coord - self.min_img_coord) + 1).astype(np.int32)  # [h, w]
        
        # Process the original dataset
        if (not self.preload_data_path is None) and os.path.exists(self.preload_data_path + '_PCs.pth'):
            self.list_of_PCs = self.dataset.load_data(self.preload_data_path + '_PCs.pth')
            self.list_of_BBs = self.dataset.load_data(self.preload_data_path + '_BBs.pth')
        else :
            self.list_of_PCs = []
            self.list_of_BBs = []
            for tracklet in tqdm(self.list_of_tracklet_anno, ncols=opts.ncols, \
                                ascii=True, desc="Create %s PCs & BBs"%split.split('_')[1]):
                for idx in range(len(tracklet)):
                    this_PC, this_BB = self.getBBandPC(tracklet[idx])
                    this_PC = kitti_utils.cropPC(this_PC, this_BB, offset=10, scale=1.0) 
                    self.list_of_PCs.append(this_PC)
                    self.list_of_BBs.append(this_BB)   
            self.dataset.save_data(self.preload_data_path + '_PCs.pth', self.list_of_PCs)
            self.dataset.save_data(self.preload_data_path + '_BBs.pth', self.list_of_BBs)
        
        for i in tqdm(range(len(self.list_of_tracklet_anno)),  ncols=opts.ncols):
            list_of_anno = self.list_of_tracklet_anno[i]
            cnt = 0
            for anno in list_of_anno:
                anno["model_idx"] = i
                anno["relative_idx"] = cnt
                cnt += 1   
        
    def __getitem__(self, index):
        return self.getitem(index)
    
    def getPCandBBfromIndex(self, anno_idx):
        this_PC = self.list_of_PCs[anno_idx]
        this_BB = self.list_of_BBs[anno_idx]
        return this_PC, this_BB
    
    def getitem(self, index):
        anno_idx = self.getAnnotationIndex(index)
        sample_idx = self.getSearchSpaceIndex(index)
        
        this_anno = self.list_of_anno[anno_idx]
        
        # 1. Get the search point cloud
        cur_PC, cur_BB = self.getPCandBBfromIndex(anno_idx)
        
        if sample_idx == 0:
            # sample_offsets = np.zeros(4)
            gaussian = searchspace.KalmanFiltering(bnd=[0.1, 0.1, 0.05, 0.5])
        else:
            gaussian = searchspace.KalmanFiltering(bnd=[1, 1, 0.5, 5])
        sample_offsets = gaussian.sample(1)[0]
        sample_cur_BB = kitti_utils.getOffsetBB_data(cur_BB, sample_offsets)
        
        target_PC, tgt_surf_cla, tgt_offcenter, adapt_cur_BB = \
            kitti_utils.cropAndCenterPC_withClaAndOff(cur_PC, sample_cur_BB, cur_BB, offset=self.offset_BB, scale=1.25, limit_area=self.area_extents)
        
        if target_PC.nbr_points() <= self.min_points_num:   
            return self.getitem(np.random.randint(0, self.__len__()))

        target_PC = target_PC.points
        target_PC, tgt_surf_cla, tgt_offcenter = kitti_utils.subsamplePC_withClaAndOff(target_PC, tgt_surf_cla, tgt_offcenter, self.subsample_number)
        
        # 2. Get the template point cloud
        if this_anno["relative_idx"] == 0:
            prev_idx = anno_idx
            fir_idx = anno_idx
        else:
            prev_idx = anno_idx - 1
            fir_idx = anno_idx - this_anno["relative_idx"]
        pre_PC, pre_BB = self.getPCandBBfromIndex(prev_idx)
        fir_PC, fir_BB = self.getPCandBBfromIndex(fir_idx)
        
        if sample_idx == 0:
            samplegt_offsets = np.zeros(4)
        else:
            samplegt_offsets = np.random.uniform(low=-0.3, high=0.3, size=4)
            samplegt_offsets[3] = samplegt_offsets[3]*5.0
        sample_pre_BB= kitti_utils.getOffsetBB_data(pre_BB, samplegt_offsets)

        model_PC = kitti_utils.getModel([fir_PC, pre_PC], [fir_BB, sample_pre_BB], scale=1.25)
        
        if model_PC.nbr_points() <= self.min_points_num:
            return self.getitem(np.random.randint(0, self.__len__()))

        model_PC = np.array(model_PC.points, dtype=np.float32)
        model_PC = kitti_utils.subsamplePC(model_PC, self.subsample_number//2)

        # 3. Set image grid extents
        offcenter = tgt_offcenter[0].numpy()
        corners = adapt_cur_BB.corners()[:2].T 
        corners = corners[[0,2,4,6]]

        output_h = self.img_size[1]   
        output_w = self.img_size[0]   

        # hot_map
        corners_int = np.floor((corners / self.xy_size) - self.min_img_coord).astype(np.int32)    
        corners_int_ul = np.min(corners_int,axis=0)                          
        corners_int_br = np.max(corners_int,axis=0)                 
        x = np.arange(corners_int_ul[0], corners_int_br[0]+1, 1)    
        y = np.arange(corners_int_ul[1], corners_int_br[1]+1, 1)    
        xx, yy = np.meshgrid(x,y)                                   

        hot_map_grid = np.concatenate([xx[:,:,np.newaxis], yy[:,:,np.newaxis]], axis=2)  
        ct = offcenter[:2]                                  
        ct_image = (ct / self.xy_size) - self.min_img_coord 
        ct_image_int = np.floor(ct_image).astype(np.int32)  

        # (local_h, local_w)
        hot_map_grid = np.sqrt(np.sum((hot_map_grid-ct_image_int)**2, axis=2))  
        hot_map_grid[hot_map_grid==0] = 1e-6                                    
        hot_map_grid = 1.0 / hot_map_grid                                       
        # (1, h, w)
        hot_map = np.zeros((1, output_h, output_w), dtype=np.float32)           
        # center: 1.0   around: 0.8     else: 0.0
        if hot_map[0,corners_int_ul[1]:corners_int_br[1]+1,corners_int_ul[0]:corners_int_br[0]+1].shape!=(corners_int_br[1]+1-corners_int_ul[1],corners_int_br[0]+1-corners_int_ul[0]):
            return self.getitem(np.random.randint(0, self.__len__()))
        hot_map[0, corners_int_ul[1]:corners_int_br[1]+1, corners_int_ul[0]:corners_int_br[0]+1] = hot_map_grid
        hot_map[0, ct_image_int[1], ct_image_int[0]] = 1.0      
        hot_map[0, [ct_image_int[1], ct_image_int[1], ct_image_int[1]+1, ct_image_int[1]-1], \
                [ct_image_int[0]-1, ct_image_int[0]+1, ct_image_int[0], ct_image_int[0]]] = 0.8   

        # ((2r+1)^2,3) x,y,ry
        local_offsets = np.zeros(((2*self.regress_radius+1)**2, 3), dtype=np.float32)    
        # (1,1)
        z_axis = np.array([[offcenter[2]]], dtype=np.float32)                       
        # center index
        index_center = np.array([ct_image_int[1]*output_w + ct_image_int[0]], dtype=np.int64)   
        index_offsets = np.zeros(((2*self.regress_radius+1)**2,), dtype=np.int64)   
        for i in range(-self.regress_radius, self.regress_radius+1):     
            for j in range(-self.regress_radius, self.regress_radius+1): 
                offsets = np.zeros((3,), dtype=np.float32)
                offsets[:2] = ct_image - ct_image_int - np.array([i,j])  
                # rotate
                offsets[2] = offcenter[3]  
                local_offsets[(i+self.regress_radius)*(2*self.regress_radius+1)+(j+self.regress_radius)] = offsets   
                ind_int = ct_image_int + np.array([i,j])    
                index_offsets[(i+self.regress_radius)*(2*self.regress_radius+1)+(j+self.regress_radius)] = ind_int[1]*output_w + ind_int[0]  

        hot_map = torch.from_numpy(hot_map)                 # (1, H, W)
        index_center = torch.from_numpy(index_center)       # (1, )
        z_axis = torch.from_numpy(z_axis)                   # (1, 1)
        index_offsets = torch.from_numpy(index_offsets)     # ((2*degress_ratio+1)**2,  ) ==> (25, )
        local_offsets = torch.from_numpy(local_offsets)     # ((2*degress_ratio+1)**2, 3) ==> (25, 3)
            
        return  {
            'template_pc':      model_PC,
            'search_pc':        target_PC,
            'heat_map':         hot_map,
            'index_center':     index_center,
            'z_axis':           z_axis,
            'index_offsets':    index_offsets,
            'local_offsets':    local_offsets,
        }

    def __len__(self):
        return len(self.list_of_anno) * self.num_candidates_perframe

    def getAnnotationIndex(self, index):
        return int(index / (self.num_candidates_perframe))

    def getSearchSpaceIndex(self, index):
        return int(index % self.num_candidates_perframe)

class TestDataset(BaseDataset):
    def __init__(self, opts, split):
        super(TestDataset, self).__init__(
            which_dataset = opts.which_dataset,
            path = opts.db.data_dir,
            split = split,
            category_name = opts.db.category_name,
            offset_BB = opts.offset_BB,
            scale_BB = opts.scale_BB
        )
        # common parameters
        self.which_dataset = opts.which_dataset
        self.split = split
        self.subsample_number = opts.subsample_number
        self.offset_BB = opts.offset_BB
        self.scale_BB = opts.scale_BB
        
        self.voxel_size=np.array(opts.voxel_size)
        self.xy_size=np.array(opts.xy_size) * opts.downsample
        self.area_extents = np.array(opts.area_extents).reshape(3, 2)
        self.xy_area_extents = np.array(opts.xy_area_extents).reshape(2, 2)

        extents_transpose = np.array(self.xy_area_extents).transpose()
        assert extents_transpose.shape == (2, 2), "Extents are the wrong shape {}".format(extents_transpose.shape)

        # image grid extents
        self.min_img_coord = np.floor(extents_transpose[0] / self.xy_size)
        voxel_extents_transpose = self.area_extents.transpose()
        self.scene_ground = voxel_extents_transpose[0]
        self.voxel_grid_size = (np.ceil(voxel_extents_transpose[1] / self.voxel_size) -
                                np.floor(voxel_extents_transpose[0] / self.voxel_size)).astype(np.int32)

    def getitem(self, index):
        list_of_anno = self.list_of_tracklet_anno[index]
        PCs = []
        BBs = []
        for anno in list_of_anno:
            this_PC, this_BB = self.getBBandPC(anno)
            PCs.append(this_PC)
            BBs.append(this_BB)
        return PCs, BBs, list_of_anno

    def __len__(self):
        return len(self.list_of_tracklet_anno)
    
    def get_tracklet_framse(self):
        tracklet_length = list(map(lambda x:len(x), self.list_of_tracklet_anno))

        return sum(tracklet_length)

class TestDataset_WOD():
    def __init__(self, opts, pc_type='raw_pc'):
        # pc_type: raw_pc, ground_pc, clean_pc
        self.tracklet_id = opts.db.tracklet_id
        self.segment_name = opts.db.segment_name
        self.data_folder = opts.db.data_dir
        
        self.start_frame = opts.db.frame_range[0]
        self.end_frame = opts.db.frame_range[1]
        
        self.cur_frame = 0
        self.tracklet_lenth = self.end_frame - self.start_frame
        
        self.gt_info = np.load(os.path.join(self.data_folder, 'gt_info', '{:}.npz'.format(self.segment_name)), allow_pickle=True)
        self.pcs = np.load(os.path.join(self.data_folder, 'pc', pc_type, '{:}.npz'.format(self.segment_name)), allow_pickle=True)
        
        self.PCs, self.BBs = self._create_pcs_and_bboxs()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.cur_frame > self.tracklet_lenth:
            raise StopIteration

        cur_pc = self.PCs[self.cur_frame]
        cur_bbox = self.BBs[self.cur_frame]

        self.cur_frame += 1
        return cur_pc, cur_bbox
    
    def _create_pcs_and_bboxs(self):
        PCs = []
        BBs = []
        
        for idx in range(self.start_frame, self.end_frame + 1):
            this_PC, this_BB = self._getBBandPC(idx)
            PCs.append(this_PC)
            BBs.append(this_BB)
        
        return PCs, BBs
    
    def _getBBandPC(self, idx):
        this_PC = PointCloud(self.pcs[str(idx)].T)
        
        frame_bboxes = self.gt_info['bboxes'][idx]
        frame_ids = self.gt_info['ids'][idx]
        index = frame_ids.index(self.tracklet_id)
        bbox = frame_bboxes[index]
        
        center = [bbox[0], bbox[1], bbox[2]] 
        size = [bbox[5], bbox[4], bbox[6]]
        orientation = Quaternion(axis=[0, 0, 1], angle=bbox[3])
        
        this_BB = BoundingBox(center, size, orientation)
        
        return this_PC, this_BB
    
    def get_instance_lenth(self):
        return self.BBs[0].wlh[1]
    
    def get_tracklet_lenth(self):
        return len(self.BBs)
    
    def get_PCs_and_BBs(self):
        return self.PCs, self.BBs