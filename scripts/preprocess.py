# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 13:51:03 2020

@author: TSAI, TUNG-CHEN
@update: 2021/10/06
@inputs: audio.wav
@saved dtype: image.png in uint16
"""

DIRECTORY = r"../dataset/audio/"
WALK = True
SAVE_DIRECTORY = r"../dataset/preprocessed/"

LABEL_TYPE = 'index'
PLOT = 3
RENDER = True

from wtbd.preprocess import preprocess
# =============================================================================
# 
# =============================================================================
if __name__ == '__main__':
    data = preprocess(DIRECTORY, 
                      walk=WALK, 
                      save_directory=SAVE_DIRECTORY, 
                      label_type=LABEL_TYPE, 
                      plot=PLOT, 
                      render=RENDER)


