import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np

# all BGR sequence in C

def compute_flow_liteflownet3(imgL, imgR, model):
    '''
    imgL: (1,3,h,w) Tensor [0, 255]
    return (1,2,h,w) Tensor flow
    '''
    model.eval()

    intWidth = imgL.shape[3]
    intHeight = imgL.shape[2]

    import math
    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

    nimgL = F.interpolate(input=imgL, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False) / 255.0
    nimgR = F.interpolate(input=imgR, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False) / 255.0

    model.training = False
    with torch.no_grad():
        outputs = model(nimgL.cuda(), nimgR.cuda())
        flow0 = outputs[0][0]
        flow0 = flow0.detach().cpu()

    flow = F.interpolate(input=flow0, size=(intHeight, intWidth), mode='bilinear', align_corners=False)
    flow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    flow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    return flow


def compute_flow_pwcnet(imgL, imgR, model, _s=16):
    '''
    imgL: (1,3,h,w) Tensor [0, 255]
    '''
    model.eval()
    _,_,h,w = imgL.shape
    s = _s  # for horfnet, 16 is preferred
    if imgL.shape[2] % s != 0:
        times = imgL.shape[2]//s
        bottom_pad = (times+1)*s -imgL.shape[2]
    else:
        bottom_pad = 0

    if imgL.shape[3] % s != 0:
        times = imgL.shape[3]//s
        right_pad = (times+1)*s-imgL.shape[3]
    else:
        right_pad = 0

    imgL = F.pad(imgL,(0,right_pad,0,bottom_pad))
    imgR = F.pad(imgR,(0,right_pad,0,bottom_pad))


    # for irr-pwc net
    input_dict = {}
    input_dict['input1'] = (imgL/255.0).cuda()
    input_dict['input2'] = (imgR/255.0).cuda()
    _,_,nh,nw = imgL.shape

    model.training = False
    with torch.no_grad():
        outputs = model(input_dict)

    realflow = outputs['flow']

    # re-crop
    realflow = realflow[:,:,:h,:w]

    return realflow


def compute_flow_irrpwc(imgL, imgR, model, _s=16):
    '''
    imgL: (1,3,h,w) Tensor [0, 255]
    '''
    model.eval()
    _,_,h,w = imgL.shape
    s = _s  # for horfnet, 16 is preferred
    if imgL.shape[2] % s != 0:
        times = imgL.shape[2]//s
        bottom_pad = (times+1)*s -imgL.shape[2]
    else:
        bottom_pad = 0

    if imgL.shape[3] % s != 0:
        times = imgL.shape[3]//s
        right_pad = (times+1)*s-imgL.shape[3]
    else:
        right_pad = 0

    imgL = F.pad(imgL,(0,right_pad,0,bottom_pad))
    imgR = F.pad(imgR,(0,right_pad,0,bottom_pad))


    # for irr-pwc net
    input_dict = {}
    input_dict['input1'] = (imgL/255.0).cuda()
    input_dict['input2'] = (imgR/255.0).cuda()
    _,_,nh,nw = imgL.shape

    model.training = False
    with torch.no_grad():
        outputs = model(input_dict)

    realflow = outputs['flow']

    # re-crop
    realflow = realflow[:,:,:h,:w]

    return realflow


def compute_flow_vcn(imgL, imgR, model, _s=64):
    '''
    imgL: (1,3,h,w) Tensor [0, 255]
    '''

    mean_L = torch.Tensor([0.32825609, 0.36677923, 0.40140483]).view(1,3,1,1)
    mean_R = torch.Tensor([0.32535163, 0.36332119, 0.39736035]).view(1,3,1,1)

    model.eval()

    imgL = imgL.squeeze(0).permute(1,2,0).numpy()
    imgR = imgR.squeeze(0).permute(1,2,0).numpy()
    maxh = imgL.shape[0]
    maxw = imgR.shape[1]
    max_h = int(maxh // 64 * 64)
    max_w = int(maxw // 64 * 64)
    if max_h < maxh: max_h += 64
    if max_w < maxw: max_w += 64

    input_size = imgL.shape
    imgL = cv.resize(imgL, (max_w, max_h))
    imgR = cv.resize(imgR, (max_w, max_h))

    imgL = torch.Tensor(imgL[None]).permute(0, 3, 1, 2)
    imgR = torch.Tensor(imgR[None]).permute(0, 3, 1, 2)

    # for VCN
    imgL = imgL/255.0 - mean_L
    imgR = imgR/255.0 - mean_R
    imgLR = torch.cat([imgL, imgR], 0).cuda()

    model.training = False
#    model.training = True
    with torch.no_grad():
        outputs = model(imgLR)

    pred_disp, entropy = outputs

    # upsampling
    pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()
    pred_disp = cv.resize(np.transpose(pred_disp,(1,2,0)), (input_size[1], input_size[0]))
    pred_disp[:,:,0] *= input_size[1] / max_w
    pred_disp[:,:,1] *= input_size[0] / max_h

    realflow = torch.Tensor(pred_disp).permute(2, 0, 1).unsqueeze(0)

    return realflow


def compute_flow_vcn_mb14(imgL, imgR, model, _s=64, g_h=960, g_w=1472):
    '''
    imgL: (1,3,h,w) Tensor [0, 255]
    '''

    mean_L = torch.Tensor([0.32825609, 0.36677923, 0.40140483]).view(1,3,1,1)
    mean_R = torch.Tensor([0.32535163, 0.36332119, 0.39736035]).view(1,3,1,1)

    model.eval()

    imgL = imgL.squeeze(0).permute(1,2,0).numpy()
    imgR = imgR.squeeze(0).permute(1,2,0).numpy()
    maxh = imgL.shape[0]
    maxw = imgR.shape[1]
#    max_h = int(maxh // 64 * 64)
#    max_w = int(maxw // 64 * 64)
#    if max_h < maxh: max_h += 64
#    if max_w < maxw: max_w += 64
    max_h = g_h
    max_w = g_w

    input_size = imgL.shape
    imgL = cv.resize(imgL, (max_w, max_h))
    imgR = cv.resize(imgR, (max_w, max_h))

    imgL = torch.Tensor(imgL[None]).permute(0, 3, 1, 2)
    imgR = torch.Tensor(imgR[None]).permute(0, 3, 1, 2)

    # for VCN
    imgL = imgL/255.0 - mean_L
    imgR = imgR/255.0 - mean_R
    imgLR = torch.cat([imgL, imgR], 0).cuda()

    model.training = False
#    model.training = True
    with torch.no_grad():
        outputs = model(imgLR)

    pred_disp, entropy = outputs

    # upsampling
    pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()
    pred_disp = cv.resize(np.transpose(pred_disp,(1,2,0)), (input_size[1], input_size[0]))
    pred_disp[:,:,0] *= input_size[1] / max_w
    pred_disp[:,:,1] *= input_size[0] / max_h

    realflow = torch.Tensor(pred_disp).permute(2, 0, 1).unsqueeze(0)

    return realflow


def compute_flow_flownetS(imgL, imgR, model, _s=64):
    '''
    imgL: (1,3,h,w) Tensor [0, 255]
    '''
    model.training = False
    model.eval()
    _,_,h,w = imgL.shape

    im_mean = torch.Tensor([0.45, 0.432, 0.411]).view(1,3,1,1)

    imgL = imgL/255.0 - im_mean
    imgR = imgR/255.0 - im_mean

    s = _s  # for horfnet, 16 is preferred
    if imgL.shape[2] % s != 0:
        times = imgL.shape[2]//s
        bottom_pad = (times+1)*s -imgL.shape[2]
    else:
        bottom_pad = 0

    if imgL.shape[3] % s != 0:
        times = imgL.shape[3]//s
        right_pad = (times+1)*s-imgL.shape[3]
    else:
        right_pad = 0

    imgL = F.pad(imgL,(0,right_pad,0,bottom_pad))
    imgR = F.pad(imgR,(0,right_pad,0,bottom_pad))

    _,_,nh,nw = imgL.shape

    inputs = torch.cat([imgL, imgR], dim=1).cuda()

    with torch.no_grad():
        outputs = model(inputs)

    flow2 = outputs[0].detach().cpu()
    realflow = F.interpolate(flow2, size=(nh,nw), mode='bilinear', align_corners=False) * 20.0

    # re-crop
    realflow = realflow[:,:,:h,:w]
    return realflow


def compute_flow_raft(imgL, imgR, model, _s=64):
    '''
    imgL: (1,3,h,w) Tensor [0, 255]
    return (1,2,h,w) Tensor flow
    '''
    model.eval()
    _,_,h,w = imgL.shape

    s = _s  # for horfnet, 16 is preferred
    if imgL.shape[2] % s != 0:
        times = imgL.shape[2]//s
        bottom_pad = (times+1)*s -imgL.shape[2]
    else:
        bottom_pad = 0

    if imgL.shape[3] % s != 0:
        times = imgL.shape[3]//s
        right_pad = (times+1)*s-imgL.shape[3]
    else:
        right_pad = 0

    imgL = F.pad(imgL,(0,right_pad,0,bottom_pad))
    imgR = F.pad(imgR,(0,right_pad,0,bottom_pad))

    imgL = imgL.cuda()
    imgR = imgR.cuda()

    with torch.no_grad():
        flow_low0, flow_high0 = model(imgL, imgR, test_mode=True)
        torch.cuda.empty_cache()

    # re-crop
    flow_high0 = flow_high0[:,:,:h,:w]  # b,2,h,w
    return flow_high0


def compute_flow_egofnet(imgL, imgR, model, GAUSS_K=1.3, GAUSS_C=2.0, _s=64):
    '''
    imgL: (1,3,h,w) Tensor [0, 255]
    return (1,2,h,w) Tensor flow
    '''
    model.eval()
    _,_,h,w = imgL.shape

    s = _s  # for horfnet, 16 is preferred
    if imgL.shape[2] % s != 0:
        times = imgL.shape[2]//s
        bottom_pad = (times+1)*s -imgL.shape[2]
    else:
        bottom_pad = 0

    if imgL.shape[3] % s != 0:
        times = imgL.shape[3]//s
        right_pad = (times+1)*s-imgL.shape[3]
    else:
        right_pad = 0

    imgL = F.pad(imgL,(0,right_pad,0,bottom_pad))
    imgR = F.pad(imgR,(0,right_pad,0,bottom_pad))

    _, _, exh, exw = imgR.shape


    imgL = imgL.cuda()
    imgR = imgR.cuda()

    with torch.no_grad():
        flow_low0, flow_high0 = model(imgL, imgR, corr_moduler=None, test_mode=True)
        flow_low0 = flow_low0.detach().cpu()
        flow_high0 = flow_high0.detach().cpu()
        torch.cuda.empty_cache()

        flow_low1, flow_high1 = model(imgR, imgL, corr_moduler=None, test_mode=True)
        flow_low1 = flow_low1.detach().cpu()
        flow_high1 = flow_high1.detach().cpu()
        torch.cuda.empty_cache()

    # moduler
    from .yp_tensor_utils import get_bi_flow_occ, generate_epipolar_corr_moduler_type_i_noaffine, generate_epipolar_corr_moduler_type_ii_noaffine
    occ0, occ1 = get_bi_flow_occ(flow_high0, flow_high1)
    from .yp_egof_utils import findFundamentalMatrixWithRandomFlow
    fmat_cv = findFundamentalMatrixWithRandomFlow(flow_high0, occ0, rt=3)
    fmat_torch = torch.Tensor(fmat_cv).unsqueeze(0)

    corr_moduler = generate_epipolar_corr_moduler_type_i_noaffine(
            fmat_torch, exh, exw, scale=8,
            gauss_k=GAUSS_K,
            gauss_c=GAUSS_C,)
    corr_moduler = corr_moduler.cuda(imgL.device)

    flow_low0g, flow_high0g = model(imgL, imgR, corr_moduler,
                                    test_mode=True)
    del corr_moduler
    flow_low0g = flow_low0g.detach().cpu()
    flow_high0g = flow_high0g.detach().cpu()
    torch.cuda.empty_cache()

    corr_moduler = generate_epipolar_corr_moduler_type_ii_noaffine(
            fmat_torch, exh, exw, scale=8,
            gauss_k=GAUSS_K,
            gauss_c=GAUSS_C,)
    corr_moduler = corr_moduler.permute(0,4,5,3,1,2).cuda(imgR.device)

    flow_low1g, flow_high1g = model(imgR, imgL, corr_moduler,
                                    test_mode=True)
    del corr_moduler
    flow_low1g = flow_low1g.detach().cpu()
    flow_high1g = flow_high1g.detach().cpu()

    occ0g, occ1g = get_bi_flow_occ(flow_high0g, flow_high1g)

    # re-crop
    flow_high0 = flow_high0[:,:,:h,:w] # b,2,h,w
    flow_high1 = flow_high1[:,:,:h,:w] # b,2,h,w
    flow_high0g = flow_high0g[:,:,:h,:w]  # b,2,h,w
    flow_high1g = flow_high1g[:,:,:h,:w]  # b,2,h,w
    occ0 = occ0[:,:h,:w]
    occ0g = occ0g[:,:h,:w]
    return flow_high0, flow_high1, flow_high0g, flow_high1g, occ0, occ0g, fmat_torch


def compute_flow_opencv_farneback(imgL, imgR, model=None):
    '''
    imgL: (1,3,h,w) Tensor [0, 255]
    '''
    imgL = imgL.squeeze(0).permute(1,2,0).numpy()
    imgR = imgR.squeeze(0).permute(1,2,0).numpy()
    im1y = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    im2y = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
    pred_flow = cv.calcOpticalFlowFarneback(im1y,im2y, None, 0.5, 8, 15, 3, 5, 1.2, 0)
    realflow = torch.Tensor(pred_flow).permute(2, 0, 1).unsqueeze(0)  # back to b,2,h,w
    return realflow


def compute_flow_opencv_tvl1(imgL, imgR, model=None, scale=0.5):
    '''
    imgL: (1,3,h,w) Tensor [0, 255]
    '''
    imgL = imgL.squeeze(0).permute(1,2,0).numpy()
    imgR = imgR.squeeze(0).permute(1,2,0).numpy()
    im1y = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    im2y = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
    im1y = cv.resize(im1y, None, fx=scale, fy=scale)
    im2y = cv.resize(im2y, None, fx=scale, fy=scale)
    tvl1 = cv.optflow.DualTVL1OpticalFlow_create()  # init
    pred_flow = tvl1.calc(im1y, im2y, None)
    pred_flow_x = 1/scale*cv.resize(pred_flow, None, fx=1/scale, fy=1/scale)
    realflow = torch.Tensor(pred_flow_x).permute(2, 0, 1).unsqueeze(0)  # back to b,2,h,w
    return realflow