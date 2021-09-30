# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:19:24 2020

@author: TSAI, TUNG-CHEN
@update: 2021/09/19
"""

import os
import psutil
# =============================================================================
### Functions
# =============================================================================
def memory_check(memory_limit, unit='GB'):
    unit_conversion = {
        "GB": 1e+09, 
        "MB": 1e+06, 
        "KB": 1e+03, 
        "B": 1
    }
    
    coefficient = unit_conversion.get(unit, None)
    if coefficient is None:
        raise ValueError(
            "Invalid `unit`: \'{0}\'! Should be one of: {1}".format(
            unit, list(unit_conversion.keys()))
        )
    
    process = psutil.Process(os.getpid())
    memusage = process.memory_info().rss / coefficient    # GB
    
    if memusage > memory_limit:    # GB
        raise MemoryError(
            "Memory usage is larger than "\
            "`memory_limit` ({0} {1})! ".format(memory_limit, unit) + \
            "Please check the data size or set a higher `memory_limit`."
        )


