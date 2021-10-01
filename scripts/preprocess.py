# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 13:51:03 2020

@author: TSAI, TUNG-CHEN
@update: 2021/10/01
@inputs: audio.wav
@saved dtype: image.png in uint16
"""

DIRECTORY = r"../dataset/audio/"
WALK = True
SAVE_DIRECTORY = r"../dataset/preprocessed/"

PLOT = 3
RENDER = True

from wtbd.preprocessors import Preprocessor, check_for_remove
# =============================================================================
# ---- main
# =============================================================================
if __name__ == '__main__':
    if SAVE_DIRECTORY is not None:
        check_for_remove(SAVE_DIRECTORY)
    
    preprocessor = Preprocessor(
        mode='dynamic', 
        label_type='index', 
        SPL=True, 
        shape=[96, 96], 
        begin=30, 
        cutin_freq=4000, 
        cutoff_freq=None, 
        plot=PLOT, 
        render=RENDER, 
        print_fn=print
    )
    
    results = preprocessor(
        directory=DIRECTORY, 
        walk=WALK, 
        save_dir=SAVE_DIRECTORY, 
        sensitivity=None, 
        scale=None, 
        rslimits=None
    )

