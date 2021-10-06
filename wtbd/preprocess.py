# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 13:51:03 2020

@author: TSAI, TUNG-CHEN
@update: 2021/10/06
@inputs: audio.wav
@saved dtype: image.png in uint16
"""

from .preprocessors import Preprocessor, check_for_remove
# =============================================================================
# 
# =============================================================================
def preprocess(directory, 
               walk=True, 
               save_directory=None, 
               label_type=None, 
               plot=3, 
               render=True):
    if save_directory is not None:
        check_for_remove(save_directory)
    
    preprocessor = Preprocessor(
        mode='dynamic', 
        SPL=True, 
        shape=[96, 96], 
        begin=30, 
        cutin_freq=4000, 
        cutoff_freq=None, 
        label_type=label_type, 
        plot=plot, 
        render=render, 
        print_fn=print
    )
    
    results = preprocessor(
        directory=directory, 
        walk=walk, 
        save_directory=save_directory, 
    )
    
    return results


