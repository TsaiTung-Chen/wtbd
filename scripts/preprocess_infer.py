#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 18:11:58 2021

@author: TSAI, TUNG-CHEN
@update: 2021/10/05
"""

MODEL_NAME = 'PhysicalCNN'

DIRECTORY = r"../dataset/audio/"
WALK = True
PLOT = 1

SUBSET = 'all'


from wtbd.infer import infer
from wtbd.utils import print_info
from wtbd.preprocess import preprocess
from wtbd.data_collectors import fetch
# =============================================================================
# 
# =============================================================================
def preprocess_infer(modelname, directory, walk=True, plot=1, subset='all'):
    data = preprocess(directory, walk=walk, plot=plot)
    data = fetch(data, subset=subset)
    print()
    print_info(data['info'])
    
    results = infer(modelname, data)
    
    return data, results


# =============================================================================
# 
# =============================================================================
if __name__ == '__main__':
    data, results = preprocess_infer(MODEL_NAME, 
                                     DIRECTORY, 
                                     walk=WALK, 
                                     plot=PLOT, 
                                     subset=SUBSET)


