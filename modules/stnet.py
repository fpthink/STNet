import pdb
import torch
import torch.nn as nn
from collections import namedtuple
import torch.nn.functional as F

from modules.backbone_net import Pointnet_Backbone
from modules.attention import corss_attention, local_self_attention

from modules.voxel_utils.voxel.voxelnet import Conv_Middle_layers
from modules.voxel_utils.voxel.region_proposal_network import RPN
from modules.voxel_utils.voxelization import Voxelization

class STNet_Tracking(nn.Module):
    def __init__(self, opts):
        super(STNet_Tracking, self).__init__()
        # ---------------------------- Param --------------------------
        self.opts = opts
        
        self.voxel_size = opts.voxel_size
        self.voxel_area = opts.voxel_area
        self.scene_ground = opts.scene_ground
        self.min_img_coord = opts.min_img_coord
        self.xy_size = opts.xy_size
        
        self.mode = opts.mode
        self.feat_emb = opts.feat_emb
        self.iters = opts.iters
        self.attention_type = 'linear'
        self.knn_num = opts.knn_num
        
        # -------------------------- Backbone -------------------------
        self.backbone_net = Pointnet_Backbone(opts.n_input_feats, use_xyz=opts.use_xyz)
        
        # -------------------------- Attention ------------------------
        self.cross_stage1 = corss_attention(self.feat_emb, self.iters, self.attention_type)                     
        self.local_stage1 = local_self_attention(self.feat_emb, self.iters, self.attention_type, self.knn_num)  

        self.cross_stage2 = corss_attention(self.feat_emb, self.iters, self.attention_type)            
        self.local_stage2 = local_self_attention(self.feat_emb, self.iters, self.attention_type, self.knn_num)   
        
        # -------------------------- Detection ------------------------
        self.voxelize = Voxelization(
                                self.voxel_area[0], 
                                self.voxel_area[1], 
                                self.voxel_area[2], 
                                scene_ground = self.scene_ground, 
                                mode = self.mode, 
                                voxel_size = self.voxel_size)
        self.cml = Conv_Middle_layers(inplanes=3+32)
        self.RPN = RPN()
        
    def xcorr(self, search_feat, search_xyz, template_feat, template_xyz):       
        search_feat1_a = self.cross_stage1(search_feat, search_xyz, template_feat, template_xyz)
        search_feat1_b = self.local_stage1(search_feat1_a, search_xyz)
        search_feat2_a = self.cross_stage2(search_feat1_b, search_xyz, template_feat, template_xyz)
        search_feat2_b = self.local_stage2(search_feat2_a, search_xyz)

        return search_feat2_b    
        
    def forward(self, template, search):
        """
        template:   [B, 512, 3] or [B, 512, 6]
        search:     [B, 1024, 3] or [B, 1024, 6]

        """
        # ---------------------- Siamese Network ----------------------
        template_xyz, template_feature = self.backbone_net(template, [256, 128, 64])
        search_xyz, search_feature = self.backbone_net(search, [512, 256, 128])


        # -------------------- correlation learning  ---------------
        fusion_feature = self.xcorr(search_feature, search_xyz, template_feature, template_xyz)

        # ---------------------- Detection ----------------------
        fusion_xyz_feature = torch.cat((search_xyz.transpose(1, 2).contiguous(), fusion_feature), dim = 1)
        voxel_features = self.voxelize(fusion_xyz_feature, search_xyz)
        voxel_features = voxel_features.permute(0, 1, 4, 3, 2).contiguous()
        cml_out = self.cml(voxel_features)
        pred_hm, pred_loc, pred_z_axis = self.RPN(cml_out)

        return pred_hm, pred_loc, pred_z_axis
    