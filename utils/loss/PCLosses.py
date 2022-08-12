import torch
import torch.nn as nn


# Calculate the pairwise distance between point clouds in a batch
def batch_pairwise_dist(x, y, use_cuda=True):
    x = x.transpose(2, 1)
    y = y.transpose(2, 1)
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    if use_cuda:
        dtype = torch.cuda.LongTensor
    else:
        dtype = torch.LongTensor
    diag_ind_x = torch.arange(0, num_points_x).cuda()
    diag_ind_y = torch.arange(0, num_points_y).cuda()
    rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(
        zz.transpose(2, 1))
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    # torch.cuda.empty_cache()
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P


# Calculate Chamfer Loss
class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts,idx):
        # preds & gts of size (BS, 3, N)

        bs,points_dim,num_points_x,seeds = preds.size()
        if idx is not None:
            num = idx.size(0)
            preds=preds.permute(0,3,1,2).contiguous()[idx[:,0],idx[:,1]]
            gts=gts.unsqueeze(-1).expand(bs,points_dim,num_points_x,seeds).permute(0,3,1,2).contiguous()[idx[:,0],idx[:,1]]
        elif idx is None:
            preds=preds.permute(0,3,1,2).contiguous().view(bs*seeds,points_dim,num_points_x)
            gts = gts.unsqueeze(-1).expand(bs, points_dim, num_points_x, seeds).permute(0, 3, 1, 2).contiguous().view(bs*seeds,points_dim,num_points_x)
            num = preds.size(0)

        P = batch_pairwise_dist(preds, gts, self.use_cuda)
        # P of size (BS, 3, N)
        mins1, _ = torch.min(P, 1)
        # mins1=mins1.view(bs,-1,num_points_x)
        loss_1 = torch.sum(mins1)  # sum of all batches
        mins2, _ = torch.min(P, 2)
        # mins2 = mins2.view(bs, -1, num_points_x)
        loss_2 = torch.sum(mins2)  # sum of all batches

        return (loss_1 + loss_2)/num


# Calculate accuracy and completeness between two point clouds
def acc_comp(preds, gts, rho=0.02):
    P = batch_pairwise_dist(preds, gts).abs().sqrt()
    pred_mins, _ = torch.min(P, 2)
    gts_mins, _ = torch.min(P, 1)
    acc = pred_mins.mean(dim=1, dtype=torch.float)
    comp = gts_mins.mean(dim=1, dtype=torch.float)
    return acc, comp
