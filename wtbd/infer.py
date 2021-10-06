#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 18:11:58 2021

@author: TSAI, TUNG-CHEN
@update: 2021/10/05
"""

import numpy as np

from .nn import get_network
from .utils import get_inputs, print_prediction
# =============================================================================
# 
# =============================================================================
def infer(modelname: str, data: dict) -> np.ndarray:
    inputs = get_inputs(modelname, data)
    
    network = get_network(modelname)
    results = network.infer(inputs)
    
    print('\nPrediction:')
    print_prediction(data, results, label_type='name')
    
    return results


