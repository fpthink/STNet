import torch
import torch.nn as nn
from modules.voxel_utils.functional.voxelization import avg_voxelize,favg_voxelize

__all__ = ['Voxelization']


class Voxelization(nn.Module):
    def __init__(self, x,y,z,scene_ground=torch.tensor([-5.6, -3.6, -2.4]),voxel_size=torch.tensor([0.3, 0.3, 0.2]),mode=True):
        super().__init__()
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)
        self.scene_ground=scene_ground
        self.voxel_size=voxel_size
        self.min_voxel_coord=torch.floor(self.scene_ground/ self.voxel_size)
        self.resolution=(-2*self.min_voxel_coord).int()
        self.mode=mode

    def forward(self, features, coords):
        coords_detach = coords.detach()
        discrete_pts = torch.floor(coords_detach / self.voxel_size.cuda())
        voxel_indices = (discrete_pts - self.min_voxel_coord.cuda()).int()
        voxel_indices=voxel_indices.transpose(1, 2).contiguous()
        if self.mode:
            return favg_voxelize(features, voxel_indices, self.x,self.y,self.z)
        else:
            return avg_voxelize(features, voxel_indices, self.x, self.y, self.z)

    def extra_repr(self):
        print('information:x {} y {} z {} min_voxel_coord {} voxel_size {} '.format(self.x,self.y,self.z,self.min_voxel_coord,self.voxel_size))
