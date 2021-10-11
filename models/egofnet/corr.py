import torch
import torch.nn.functional as F
#from utils.utils import bilinear_sampler, coords_grid
from .utils import bilinear_sampler, coords_grid

#import utils.yp_tensor_utils as ytu

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrBlock:
    def __init__(self, fmap1, fmap2, corr_moduler, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2, corr_moduler)

        if 0:  # debug
            corr_np = corr[0].detach().cpu().numpy()
            import numpy as np
            np.save('corr1_modulated.npy', corr_np)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dy = torch.linspace(-r, r, 2*r+1)
#            dy = torch.linspace(0,0,1)  # yp-debug
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)  #[9,9,2],[y,x,2],for sampling

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
#            delta_lvl = delta.view(1, 1, 2*r+1, 2)  # yp-debug
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)  # original
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)  # use bilinear to get corr volume with dx dy
            corr = corr.view(batch, h1, w1, -1)  # [b,h,w,(2r+1)**2]
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2, corr_moduler):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd)

        corr = torch.matmul(fmap1.transpose(1,2), fmap2)  # matrix mul to generate HxWxHxW size
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        corr = corr / torch.sqrt(torch.tensor(dim).float())  # (b, nh, nw, 1, nh, nw)

        ###  BEGIN GAUSSIAN MODULATION  ###
        # fmap1,2 is H/8 by W/8
        # hint_mask: (b, 2, H/8, W/8) left and right view in dim=1
        # hint_value1: (b, 2, H/8, W/8) u, v coordinate in dim=1
        # hint_value2: (b, 2, H/8, W/8)
# =============================================================================
#         corr_moduler = ytu.generate_corr_moduler_type_i(
#                 hint_mask, hint_value1, hint_value2,
#                 gauss_k=10, gauss_c=1)
# =============================================================================

#        corr = corr * corr_moduler.cuda(fmap1.device)
        if corr_moduler is not None:
            corr = corr * corr_moduler
        ###  END GAUSSIAN MODULATION    ###

        return corr


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())
