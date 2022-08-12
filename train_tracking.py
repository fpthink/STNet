import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from utils.loss.losses import RegL1Loss, FocalLoss
from utils.loss.PCLosses import ChamferLoss

from datasets.get_stnet_db import get_dataset
from modules.stnet import STNet_Tracking
from utils.show_line import print_info
from trainers.trainer import train_model, valid_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def train_tracking(opts):
    ## Init
    print_info(opts.ncols, 'Start')
    set_seed(opts.seed)
    
    ## Define dataset
    print_info(opts.ncols, 'Define dataset')
    train_loader, train_db = get_dataset(opts, partition="Train", shuffle=True)
    valid_loader, valid_db = get_dataset(opts, partition="Valid", shuffle=False)
    
    opts.voxel_size = torch.from_numpy(train_db.voxel_size.copy()).float()
    opts.voxel_area = train_db.voxel_grid_size
    opts.scene_ground = torch.from_numpy(train_db.scene_ground.copy()).float()
    opts.min_img_coord = torch.from_numpy(train_db.min_img_coord.copy()).float()
    opts.xy_size = torch.from_numpy(train_db.xy_size.copy()).float()
    
    ## Define model
    print_info(opts.ncols, 'Define model')
    model = STNet_Tracking(opts)
    if (opts.n_gpus > 1) and (opts.n_gpus >= torch.cuda.device_count()):
        model = torch.nn.DataParallel(model, range(opts.n_gpus))
    model = model.to(opts.device)
    
    ## Define optim & scheduler
    print_info(opts.ncols, 'Define optimizer & scheduler')
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate, betas=(0.9, 0.999))
    
    if opts.which_dataset.upper() == "NUSCENES":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.2)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    
    ## Define loss
    print_info(opts.ncols, 'Define loss')
    criternions = {
        'hm':         FocalLoss().to(opts.device),
        'loc':        RegL1Loss().to(opts.device),
        'z_axis':     RegL1Loss().to(opts.device),
    }
    
    ## Training
    print_info(opts.ncols, 'Start training!')

    best_loss = 9e99
    for epoch in range(1, opts.n_epoches+1):
        print('Epoch', str(epoch), 'is training:')

        # train current epoch
        train_loss = train_model(opts, model, train_loader, optimizer, criternions, epoch)
        valid_loss = valid_model(opts, model, valid_loader, criternions, epoch)

        # save current epoch state_dict
        torch.save(model.state_dict(), os.path.join(opts.results_dir, "netR_" + str(epoch) + ".pth"))
        
        # save best model state_dict
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(opts.results_dir, "Best.pth"))

        # update scheduler
        scheduler.step(epoch)

        print('======>>>>> Train: loss: %.5f, Valid: loss: %.5f <<<<<======'%(train_loss, valid_loss))
    