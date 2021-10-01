# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 17:37:18 2020

@author: TSAI, TUNG-CHEN
@update: 2021/10/01
@pipeline:
    1.
"""

import numpy as np
# =============================================================================
# 
# =============================================================================
def assert_1d(x):
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    
    if (x.ndim < 1) or ( x.shape == (x.size, 1) ):
        x = x.reshape(-1)
    
    assert x.ndim == 1
    
    return x



def binary_classification(y, threshold=0.5, keepdims=True):
    y = assert_1d(y)
    classes = (y > threshold).astype(int)    # values on interval [0, 1]
    
    if keepdims:
        return classes
    return 0 if classes.mean() < 0.5 else 1


