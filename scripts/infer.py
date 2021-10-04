#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 18:11:58 2021

@author: TSAI, TUNG-CHEN
@update: 2021/10/04
"""

MODEL_NAME = 'PhysicalCNN'

DIRECTORY = r"../dataset/preprocessed/data/"
WALK = True

SUBSET = 'all'

PLOT = 1


from wtbd.nn import get_network
from wtbd.preprocessors import Preprocessor
from wtbd.data_collectors import SubsetDataCollector, fetch
from wtbd.utils import get_inputs, print_info, show_prediction
# =============================================================================
# 
# =============================================================================
def infer(modelname, 
          directory, 
          walk=False, 
          subset='all', 
          plot=1):
    try:
        data_collector = SubsetDataCollector()
        data = data_collector(directory, subset=subset)
    except FileNotFoundError:
        preprocessor = Preprocessor(plot=plot)
        data = preprocessor(directory, walk=walk)
        data = fetch(data, subset=subset)
        print()
    print_info(data['info'])
    
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
                          subset=SUBSET, 
                          plot=PLOT)


