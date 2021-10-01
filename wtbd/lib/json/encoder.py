# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:47:13 2021

@author: TSAI, TUNG-CHEN
@update: 2021/03/09
"""

import json
import datetime
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        dtype = obj.dtype.type if hasattr(obj, 'dtype') else type(obj)
        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        elif np.issubdtype(dtype, np.integer):
            return int(obj)
        
        elif np.issubdtype(dtype, np.floating):
            return float(obj)
        
        elif np.issubdtype(dtype, np.complexfloating):
            return complex(obj)
        
        elif np.issubdtype(dtype, np.bytes_):
            return bytes(obj)
        
        elif np.issubdtype(dtype, np.str_):
            return str(obj)
        
        elif np.issubdtype(dtype, np.bool_):
            return bool(obj)
        
        elif np.issubdtype(dtype, np.datetime64):
            return datetime.datetime(obj)
        
        elif np.issubdtype(dtype, np.timedelta64):
            return datetime.timedelta(obj)
        else:
            return super().default(obj)


