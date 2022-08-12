import os
import pickle

import numpy as np
import pandas as pd
from pyquaternion import Quaternion

from torch.utils.data import Dataset

from utils.data_classes import PointCloud, BoundingBox

class kittiDataset():
    def __init__(self, path, which_dataset):
        self.which_dataset = which_dataset
        self.KITTI_Folder = path
        self.KITTI_velo = os.path.join(self.KITTI_Folder, "velodyne")
        self.KITTI_label = os.path.join(self.KITTI_Folder, "label_02")
        self.KITTI_calib = os.path.join(self.KITTI_Folder, "calib")

    def getSceneID(self, split):
        if self.which_dataset.upper() == 'NUSCENES':
            if "TRAIN" in split.upper():  
                # Training SET
                if "TINY" in split.upper():
                    sceneID = [0]
                else:
                    sceneID = list(range(0, 350))
            elif "VALID" in split.upper():  
                # Validation Set
                if "TINY" in split.upper():
                    sceneID = [0]
                else:
                    sceneID = list(range(0, 10))
            elif "TEST" in split.upper():  
                # Testing Set
                if "TINY" in split.upper():
                    sceneID = [0]
                else:
                    sceneID = list(range(0, 150))
        else:
            # KITTI dataset
            if "TRAIN" in split.upper():  
                # Training SET
                if "TINY" in split.upper():
                    sceneID = list(range(0 ,1))
                else:
                    sceneID = list(range(0, 17))
            elif "VALID" in split.upper():  
                # Validation Set
                if "TINY" in split.upper():
                    sceneID = list(range(0, 1))
                else:
                    sceneID = list(range(17, 19))
            elif "TEST" in split.upper():  
                # Testing Set
                if "TINY" in split.upper():
                    sceneID = list(range(0, 1))
                else:
                    sceneID = list(range(19, 21))
            else:  
                # Full Dataset
                sceneID = list(range(21))
        return sceneID

    def getListOfAnno(self, sceneID, category_name="Car"):
        list_of_scene = [
            path for path in os.listdir(self.KITTI_velo)
            if os.path.isdir(os.path.join(self.KITTI_velo, path)) and
            int(path) in sceneID
        ]
        
        list_of_tracklet_anno = []
        for scene in list_of_scene:
            # read the label file
            label_file = os.path.join(self.KITTI_label, scene + ".txt")
            if self.which_dataset.upper()=='NUSCENES':
                df = pd.read_csv(
                    label_file,
                    sep=' ',
                    names=[
                        "frame", "track_id", "type", "truncated", "occluded",
                        "alpha", "bbox_left", "bbox_top", "bbox_right",
                        "bbox_bottom", "height", "width", "length", "x", "y", "z",
                        "rotation_y","score",'num_lidar_pts','is_key_frame'
                    ])
            else:
                df = pd.read_csv(
                    label_file,
                    sep=' ',
                    names=[
                        "frame", "track_id", "type", "truncated", "occluded",
                        "alpha", "bbox_left", "bbox_top", "bbox_right",
                        "bbox_bottom", "height", "width", "length", "x", "y", "z",
                        "rotation_y", "score"
                    ])
                
            df = df[df["type"] == category_name]
            # insert the scene dim
            df.insert(loc=0, column="scene", value=scene)
            for track_id in df.track_id.unique():
                df_tracklet = df[df["track_id"] == track_id]
                df_tracklet = df_tracklet.reset_index(drop=True)    
                tracklet_anno = [anno for index, anno in df_tracklet.iterrows()]
                list_of_tracklet_anno.append(tracklet_anno)

        return list_of_tracklet_anno

    def read_calib_file(self, filepath):
        """Read in a calibration file and parse into a dictionary."""
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                values = line.split()
                
                try:
                    ind = values[0].find(':')
                    if ind != -1:
                        data[values[0][:ind]] = np.array(
                            [float(x) for x in values[1:]]).reshape(3, 4)
                    else:
                        data[values[0]] = np.array(
                            [float(x) for x in values[1:]]).reshape(3, 4)
                except ValueError:
                    data[values[0]] = np.array(
                        [float(x) for x in values[1:]]).reshape(3, 3)
        return data

    def getBBandPC(self, anno):
        calib_path = os.path.join(self.KITTI_calib, anno['scene'] + ".txt")
        calib = self.read_calib_file(calib_path)
        # get the Tr_velo_cam matrix, which transforms the point cloud from the velo coordinate system to the cam coordinate system
        transf_mat = np.vstack((calib["Tr_velo_cam"], np.array([0, 0, 0, 1])))  # 3*4 --> 4*4
        PC, bbox = self.getPCandBBfromPandas(anno, transf_mat)
        return PC, bbox

    def getPCandBBfromPandas(self, box, calib):
        center = [box["x"], box["y"] - box["height"] / 2, box["z"]] 
        size = [box["width"], box["length"], box["height"]]
        orientation = Quaternion(axis=[0, 1, 0], radians=box["rotation_y"]) * Quaternion(axis=[1, 0, 0], radians=np.pi / 2)
        BB = BoundingBox(center, size, orientation)

        try:
            # VELODYNE PointCloud
            velodyne_path = os.path.join(self.KITTI_velo, box["scene"], f'{box["frame"]:06}.bin')
            PC = PointCloud(np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4).T)
            PC.transform(calib) 
        except :
            PC = PointCloud(np.array([[0, 0, 0]]).T)

        return PC, BB

    def load_data(self, path):
        file = open(path, "rb")
        data = pickle.load(file)
        file.close()
        return data

    def save_data(self, path, data):
        file = open(path, "wb")
        pickle.dump(data, file)
        file.close()

class BaseDataset(Dataset):
    def __init__(self, which_dataset, path, split, category_name="Car", offset_BB=np.zeros(1), scale_BB=np.ones(1)):

        self.dataset = kittiDataset(path=path, which_dataset=which_dataset)

        self.split = split
        self.category_name = category_name

        self.getBBandPC = self.dataset.getBBandPC

        self.sceneID = self.dataset.getSceneID(split=split)
        
        '''every anno include:
        "sceneID", "frame", "track_id", "type", 
        "truncated", "occluded", "alpha", 
        "bbox_left", "bbox_top", "bbox_right", "bbox_bottom", 
        "height", "width", "length", "x", "y", "z", "rotation_y"
        '''
        # list, every object is a tracklet anno
        self.list_of_tracklet_anno = self.dataset.getListOfAnno(self.sceneID, category_name)
        self.list_of_anno = [
            anno for tracklet_anno in self.list_of_tracklet_anno
            for anno in tracklet_anno
        ]

    def isTiny(self):
        return ("TINY" in self.split.upper())

    def __getitem__(self, index):
        return self.getitem(index)
