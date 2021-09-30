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
PLOT = 1

from wtbd.nn import get_network
from wtbd.utils import get_inputs
from wtbd.preprocessors import Preprocessor
from wtbd.postprocess import show_prediction
from wtbd.data_collectors import SubsetDataCollector
# =============================================================================
# 
# =============================================================================
def infer(modelname, 
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
    
    inputs = get_inputs(modelname, data)
    
    network = get_network(modelname)
    results = network.infer(inputs)
    
    print('\nPrediction:')
    show_prediction(data, results, label_type='name')
    
    return data, results


# =============================================================================
# 
# =============================================================================
if __name__ == '__main__':
    data, results = infer(MODEL_NAME, 
                          DIRECTORY, 
                          walk=WALK, 
                          sensitivity=SENSITIVITY, 
                          scale=SCALE, 
                          plot=PLOT)


