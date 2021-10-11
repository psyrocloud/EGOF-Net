# EGOF-Net
Repository for the paper "EGOF-Net: epipolar guided optical flow network for unrectified stereo matching".

Please cite the paper if you find it useful.

## Citation:
> Yunpeng Li, Baozhen Ge, Qingguo Tian, Qieni Lu, Jianing Quan, Qibo Chen, and Lei Chen, "EGOF-Net: epipolar guided optical flow network for unrectified stereo matching," Opt. Express 29, 33874-33889 (2021)

## Paper URL:
https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-29-21-33874&id=460104

# How to use:

1. Open a CMD(For windows)/Shell(For Linux) window and make sure the current working directory of python or ipython is the same as the 'demo.py' file.
2. Run the demo.py file and you can see the visualization of optical flow.
> python demo.py
3. If you want to try your image pairs, modify the 'im1' and 'im2' paths in demo.py. 


# Packages needed:
1. Python 3.7.4
2. Pytorch 1.7.0 (py37_cuda102_cudnn7_0, version later than pytorch 1.2.0 should also work)
3. Opencv 4.5.3.56 with GUI (Older version should also work, we use opencv's HighGUI to visualize our results.)
4. Numpy (Any version that is compatible with the Pytorch)

# More information
The code can work in a Windows 10 x64 PC.

The python environment is based on Anaconda 2019.10 (with python 3.7).

# The dataset and training code would be available soon.
