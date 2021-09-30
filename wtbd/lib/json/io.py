#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 21:35:58 2021

@author: TSAI, TUNG-CHEN
@update: 2021/09/25
"""

import json
from .encoder import NumpyEncoder
from ..file import insert_time_suffix, makedirs
# =============================================================================
# 
# =============================================================================
def save_json(obj, fpath, time_suffix=False, makedir=True, **kwargs):
    _kwargs_ = {"indent": 4, 
                "ensure_ascii": False, 
                "cls": NumpyEncoder}
    _kwargs_.update(kwargs)
    
    if time_suffix:
        fpath = insert_time_suffix(fpath, hyphen=False)
    if makedir:
        makedirs(fpath)
    
    with open(fpath, 'w', encoding='utf-8') as json_f:
        json.dump(obj, json_f, **_kwargs_)



def load_json(fpath, **kwargs):
    with open(fpath, 'r', encoding='utf-8') as json_f:
        return json.load(json_f, **kwargs)



def loads_json(s, **kwargs):
    return json.loads(s, **kwargs)


