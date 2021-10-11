import torch
import models

from utils import yp_compute_flow as ypcf
from utils.yp_load_state_dict import load_weights_to_model as yp_load_weights_to_model
from utils.yp_flow_viz import flow_to_image as yp_flow_to_image

from utils.yp_mkdir import ypmkdir
import numpy as np
import os

if __name__ == '__main__':
    # Step 1. load left and right images
    print('[1] Loading left and right images...')
    im1 = './image_sample/left0.png'
    im2 = './image_sample/right0.png'

    loadmodel = './pretrained_ckpt/egof-net.tar'
    savemodel = './demo_results'
    import cv2 as cv
    imgL = cv.imread(im1)
    imgR = cv.imread(im2)
    imgL = torch.Tensor(imgL).permute(2, 0, 1).unsqueeze(0)
    imgR = torch.Tensor(imgR).permute(2, 0, 1).unsqueeze(0)
    print('[1] Completed.')


    # Step 2. load pretrained weights
    print('[2] Loading pretrained weights...')
    model = models.get_model_instance_egofnet()
    model = yp_load_weights_to_model(model, loadmodel)
    model = model.cuda()
    yp_compute_flow = ypcf.compute_flow_egofnet
    print('[2] Completed.')


    # Step 3. estimate optical flow with epipolar line guidance
    print('[3] Estimating optical flow... Please wait.')
    flow_high0, flow_high1, flow_high0g, flow_high1g, occ0, occ0g, fmat_torch = \
    yp_compute_flow(imgL, imgR, model)
    print('[3] Completed.')


    # Step 4. save results
    print('[4] Saving optical flow results...')
    ypmkdir(savemodel)
    flow0g = flow_high0g.detach().cpu().squeeze(0).permute(1,2,0).numpy()
    flow1g = flow_high1g.detach().cpu().squeeze(0).permute(1,2,0).numpy()
    occ0g = occ0g.detach().cpu().squeeze(0).numpy()
    fmat = fmat_torch.detach().cpu().squeeze(0).numpy()

    imf0g = yp_flow_to_image(flow0g, convert_to_bgr=True)
    imf1g = yp_flow_to_image(flow1g, convert_to_bgr=True)
    occ0g = np.uint8(occ0g*255)

    cv.imwrite(os.path.join(savemodel, 'flow0g.png'), imf0g)
    cv.imwrite(os.path.join(savemodel, 'flow1g.png'), imf1g)
    cv.imwrite(os.path.join(savemodel, 'occ0g.png' ), occ0g)
    np.save(os.path.join(savemodel, 'flow0g.npy'), flow0g)
    np.save(os.path.join(savemodel, 'flow1g.npy'), flow1g)
    np.save(os.path.join(savemodel, 'fmat.npy'), fmat)
    im0o = cv.imread(im1)
    im1o = cv.imread(im2)
    cv.imwrite(os.path.join(savemodel, 'im0.png' ), im0o )
    cv.imwrite(os.path.join(savemodel, 'im1.png' ), im1o )
    print('[4] Saved results to "%s" directory.'%savemodel)


    # Step 5. visualization
    print('[5] Visualization.')
    cv.namedWindow('left')
    cv.namedWindow('right')
    cv.namedWindow('flow-left')
    cv.namedWindow('flow-right')
    cv.namedWindow('occlusion-left')
    cv.imshow('left', im0o)
    cv.imshow('right', im1o)
    cv.imshow('flow-left', imf0g)
    cv.imshow('flow-right', imf1g)
    cv.imshow('occlusion-left', occ0g)
    cv.waitKey(0)
    print('[5] Visualization terminated.')
