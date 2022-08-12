import os
import time
import argparse
from utils.options import opts

from utils.show_line import print_info
from train_tracking import train_tracking
from test_tracking import test_tracking

parser = argparse.ArgumentParser()
parser.add_argument('--which_dataset', type=str, default='KITTI',  help='datasets: KITT,NUSCENES,WAYMO')
parser.add_argument('--category_name', type=str, default='Car',  \
    help='KITTI:Car/Pedestrian/Van/Cyclist; nuScenes:car/pedestrian/truck/bicycle; waymo:vehicle/pedestrian/cyclist')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--n_workers', type=int, default=12, help='number of data loading workers')
parser.add_argument('--n_epoches', type=int, default=40, help='number of epochs to train for')
parser.add_argument('--n_gpus', type=int, default=2, help='# GPUs')
parser.add_argument('--train_test', type=str, default='train', help='train or test')
parser.add_argument('--model_epoch', type=int, default=30, help='which epoch model to test')
parser.add_argument('--visual', type=bool, default=False, help='save data for visualization')
# parse arguments
manual_opts = parser.parse_args()

def init_opts(opts, manual_opts):
    opts.which_dataset = manual_opts.which_dataset.upper()
    
    if opts.which_dataset.upper() not in ['KITTI', 'NUSCENES', 'WAYMO']:
        raise ValueError("Please use command '--which_dataset kitti/nuscenes/waymo' to select datasets we support.")
    
    opts.batch_size = manual_opts.batch_size
    opts.n_workers = manual_opts.n_workers
    opts.n_epoches = manual_opts.n_epoches
    opts.n_gpus = manual_opts.n_gpus
    opts.train_test = manual_opts.train_test
    opts.visual = manual_opts.visual
    
    opts.db = opts.db[opts.which_dataset]
    if opts.which_dataset.upper() == 'KITTI' and manual_opts.category_name not in ['Car', 'Pedestrian', 'Van', 'Cyclist']:
        raise ValueError("Please enter the correct species name supported by the KITTI dataset (Car/Pedestrian/Van/Cyclist).")
    if opts.which_dataset.upper() == 'NUSCENES' and manual_opts.category_name not in ['car', 'pedestrian', 'truck', 'bicycle']:
        raise ValueError("Please enter the correct species name supported by the nuScenes dataset (car/pedestrian/truck/bicycle).")
    if opts.which_dataset.upper() == 'WAYMO' and manual_opts.category_name not in ['vehicle', 'pedestrian', 'cyclist']:
        raise ValueError("Please enter the correct species name supported by the waymo open dataset (vehicle/pedestrian/cyclist).")
    opts.db.category_name = manual_opts.category_name
    
    # note that: we only use waymo oepn dataset to test the generalization ability of the kitti model
    # KITTI/WAYMO ==> kitti, NUSCENES ==> nuscenes
    # WAYMO.vehicle/pedestrian/cyclist ==> KITTI.Car/Pedestrian/Cyclist
    opts.rp_which_dataset = 'nuscenes' if opts.which_dataset.upper()=='NUSCENES' else 'kitti'   
    opts.rp_category = 'Car' if (opts.which_dataset.upper()=='WAYMO' and opts.db.category_name=='vehicle') else opts.db.category_name   
    
    opts.data_save_path = os.path.join('/opt/data/private/tracking/v2b/', ('tiny' if opts.use_tiny else 'full'), opts.rp_which_dataset)
    opts.results_dir = "./results/%s_%s" % (opts.rp_which_dataset.lower(), opts.rp_category.lower())
    
    if opts.train_test == 'train':
        opts.mode = True
        os.makedirs(opts.results_dir, exist_ok=True)
        os.makedirs(opts.data_save_path, exist_ok=True)
    elif opts.train_test == 'test':
        opts.mode = False

    opts.model_path = "%s/netR_%d.pth" % (opts.results_dir, manual_opts.model_epoch)
        
    return opts

if __name__ == '__main__':
    ## Init opts
    opts = init_opts(opts, manual_opts)
    
    ## Show the key information
    key_info = ['model_name', 'which_dataset', 'train_test', 'batch_size', 'n_epoches', \
                'rp_category', 'n_gpus', 'results_dir', 'data_save_path', 'model_path']
    print_info(opts.ncols, 'Key Info')
    for name in key_info:
        print(name, ':', opts[name])
    
    start = time.time()
    ## train or test
    if opts.train_test.lower() == 'train':
        train_tracking(opts)
    elif opts.train_test.lower() == 'test':
        test_tracking(opts)
    else:
        raise ValueError("Please use command '--train_test train/test' to select train or test model.")

    run_time = time.time() - start
    print_info(opts.ncols, 'Running time : {:.2f} min'.format(run_time/60))