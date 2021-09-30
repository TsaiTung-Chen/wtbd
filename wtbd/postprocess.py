# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 17:37:18 2020

@author: TSAI, TUNG-CHEN
@update: 2021/09/28
@pipeline:
    1.
"""

import functools
import numpy as np
from tabulate import tabulate

from .preprocessors import change_symbol
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



def show_prediction(data: dict, pred, label_type='name', print_fn=print):
    names, pred = np.squeeze(data['names']), np.squeeze(pred)
    assert len(names) == len(pred)
    
    _change_symbol = functools.partial(change_symbol, symbol_type=label_type)
    pred = list(map(_change_symbol, pred))
    index = range(len(names))
    table = tabulate(np.stack([index, names, pred], axis=-1), 
                     headers=['index', 'names', 'predicted class'])
    print_fn(table)
    
    return table


