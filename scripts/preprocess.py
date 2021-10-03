# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 13:51:03 2020

@author: TSAI, TUNG-CHEN
@update: 2021/10/03
@inputs: audio.wav
@saved dtype: image.png in uint16
"""

DIRECTORY = r"../dataset/audio/"
WALK = True
SAVE_DIRECTORY = r"../dataset/preprocessed/"

SENSITIVITY = None
SCALE = None
PLOT = 3
RENDER = True

from wtbd.preprocessors import Preprocessor, check_for_remove
# =============================================================================
# 
# =============================================================================
def preprocess(directory, 
               walk=False, 
               save_directory=None, 
               sensitivity=None, 
               scale=None, 
               plot=3, 
               render=True):
    if save_directory is not None:
        check_for_remove(save_directory)
    
    preprocessor = Preprocessor(
        mode='dynamic', 
        label_type='index', 
        SPL=True, 
        shape=[96, 96], 
        begin=30, 
        cutin_freq=4000, 
        cutoff_freq=None, 
        plot=plot, 
        render=render, 
        print_fn=print
    )
    
    results = preprocessor(
        directory=directory, 
        walk=walk, 
        save_directory=save_directory, 
        sensitivity=sensitivity, 
        scale=sensitivity, 
        rslimits=None
    )
    
    return results

# =============================================================================
# 
# =============================================================================
if __name__ == '__main__':
    data = preprocess(DIRECTORY, 
                      walk=WALK, 
                      save_directory=SAVE_DIRECTORY, 
                      sensitivity=SENSITIVITY, 
                      scale=SCALE, 
                      plot=PLOT, 
                      render=RENDER)


