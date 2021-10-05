#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 18:11:58 2021

@author: TSAI, TUNG-CHEN
@update: 2021/10/05
"""

import numpy as np

from .nn import get_network
from .utils import get_inputs, get_targets
# =============================================================================
# 
# =============================================================================
def evaluate(modelname: str, data: dict) -> np.ndarray:
    inputs = get_inputs(modelname, data)
    targets = get_targets(data)
    
    network = get_network(modelname)
    results = network.evaluate(inputs, targets)
    
    return results


