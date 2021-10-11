# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 09:53:07 2020

@author: psyrocloud
"""

import os

#def ypmkdir(outputDir):
#    if not os.path.isdir(outputDir):
#        os.mkdir(outputDir)
#    return

def ypmkdir(outputDir):
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)
    return