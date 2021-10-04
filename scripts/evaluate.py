#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 18:11:58 2021

@author: TSAI, TUNG-CHEN
@update: 2021/10/01
"""

MODEL_NAME = 'PhysicalCNN'

DIRECTORY = r"../dataset/audio/"
WALK = True

SUBSET = 'all'

PLOT = 1


from wtbd.nn import get_network
from wtbd.preprocessors import Preprocessor
from wtbd.data_collectors import SubsetDataCollector, fetch
from wtbd.utils import get_inputs, get_targets, print_info
# =============================================================================
# 
# =============================================================================
def evaluate(modelname, 
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
                             subset=SUBSET, 
                             plot=PLOT)


