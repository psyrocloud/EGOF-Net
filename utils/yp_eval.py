import torch
import numpy as np


def eval_ferr(flow_pred, flow_true, mask):
    '''Inputs are 2,H,W. Tensor, mask is H,W float tensor
    Return norm_epe, u_epe, v_epe, 1-pix_err_rate, 3-pix_err_rate, 5-pix_err_rate
    '''
    abs_diff = torch.abs(flow_pred - flow_true)
    diff_norm = torch.norm(abs_diff, p='fro', dim=0)  # h,w
    idx = mask>0  # bool

    count = torch.sum(mask)
    h, w = mask.shape
    norm_epe_all = torch.mean(diff_norm)
    norm_epe_occ = torch.mean(diff_norm[idx])

    pix3_err_all = torch.sum(diff_norm>3) / (h*w)
    pix3_err_occ = torch.sum(diff_norm[idx]>3) / (count + 1e-8)

#    p1e_idx = np.uint8(((diff_norm>1)*idx).detach().cpu().numpy())
#    p3e_idx = np.uint8(((diff_norm>3)*idx).detach().cpu().numpy())
    p3e_idx = np.uint8(((diff_norm>5)*idx).detach().cpu().numpy())

#    p5e_idx = np.uint8(((diff_norm>5)*idx).detach().cpu().numpy())

    return norm_epe_all, norm_epe_occ, pix3_err_all, pix3_err_occ, p3e_idx