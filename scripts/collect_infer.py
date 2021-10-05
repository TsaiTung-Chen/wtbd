#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 18:11:58 2021

@author: TSAI, TUNG-CHEN
@update: 2021/10/05
"""

MODEL_NAME = 'PhysicalCNN'

DIRECTORY = r"../dataset/preprocessed/data/"
WALK = True

SUBSET = 'all'


from wtbd.infer import infer
from wtbd.utils import print_info
from wtbd.data_collectors import SubsetDataCollector
# =============================================================================
# 
# =============================================================================
def collect_infer(modelname, directory, walk=True, subset='all'):
    data_collector = SubsetDataCollector()
    data = data_collector(directory, subset=subset)
    print_info(data['info'])
    results = infer(modelname, data)
    
    return data, results


# =============================================================================
# 
# =============================================================================
if __name__ == '__main__':
    data, results = collect_infer(MODEL_NAME, 
                                  DIRECTORY, 
                                  walk=WALK, 
                                  subset=SUBSET)


