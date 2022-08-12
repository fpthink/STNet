from torch.utils.data import DataLoader
from datasets.stnet_dataset import TrainDataset, TestDataset, TestDataset_WOD

def get_dataset(opts, partition, shuffle=False):
    loader, db = None, None
    
    if opts.use_tiny:
        split = "Tiny_" + partition
    else:
        split = "Full_" + partition
    
    
    if partition in ["Train", "Valid"]:
        db = TrainDataset(opts, split)
        loader = DataLoader(db, batch_size=opts.batch_size, shuffle=shuffle, num_workers=opts.n_workers, pin_memory=True)
    else:
        # Test dataset
        if opts.which_dataset.upper() in ['KITTI', 'NUSCENES']:
            db = TestDataset(opts, split)
            loader = DataLoader(db, batch_size=1, shuffle=shuffle, num_workers=opts.n_workers, pin_memory=True, collate_fn=lambda x: x)
        else:
            # waymo test
            db = TestDataset_WOD(opts, pc_type='raw_pc')

    return loader, db