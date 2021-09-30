#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 18:11:58 2021

@author: TSAI, TUNG-CHEN
@update: 2021/09/29
"""

MODEL_NAME = 'PhysicalCNN'

DIRECTORY = r"../dataset/preprocessed/data/"
WALK = True

SENSITIVITY = None
SCALE = None
PLOT = 0#???1

from wtbd.nn import get_network
from wtbd.preprocessors import Preprocessor
from wtbd.utils import get_inputs, get_targets
from wtbd.data_collectors import SubsetDataCollector
# =============================================================================
# 
# =============================================================================
def evaluate(modelname, 
             directory, 
             walk=False, 
             sensitivity=None, 
             scale=None, 
             plot=1):
    try:
        data_collector = SubsetDataCollector()
        data = data_collector(directory, subset='test')
    except FileNotFoundError:
        preprocessor = Preprocessor(plot=plot)
        data = preprocessor(directory, 
                            walk=walk, 
                            sensitivity=sensitivity, 
                            scale=scale)
        print()
    
    inputs = get_inputs(modelname, data)
    targets = get_targets(data)
    
    network = get_network(modelname)
    results = network.evaluate(inputs, targets)
    
    return data, results


# =============================================================================
# 
# =============================================================================
if __name__ == '__main__':
    data, results = evaluate(MODEL_NAME, 
                             DIRECTORY, 
                             walk=WALK, 
                             sensitivity=SENSITIVITY, 
                             scale=SCALE, 
                             plot=PLOT)


