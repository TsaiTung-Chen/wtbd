# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:19:24 2020

@author: TSAI, TUNG-CHEN
@update: 2021/05/11
"""

from decimal import Decimal, ROUND_HALF_UP
# =============================================================================
### Functions
# =============================================================================
def iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError as e:
        if 'not iterable' in str(e):
            return False
    else:
        raise


def round2int(x):
    """Round the number `x` to the nearest whole number (int).
    ex. 0.4 => 0; 0.5 => 1; and also -0.4 => 0; -0.5 => -1
    """
    return int(Decimal(str(x)).quantize(0, ROUND_HALF_UP))


def flatten(nested_iterable):
    def _flatten(nested):
        if hasattr(nested, '__iter__'):
            [ _flatten(sub) for sub in nested ]
            return
        flat.append(nested)
    #
    
    flat = list()
    [ _flatten(sub) for sub in nested_iterable ]
    
    return flat


def nested_list_map(lis: list, func) -> list:
    """Nested mapping for a nested list."""
    def _nested_fn(obj):
        if isinstance(obj, list):
            return list(map(_nested_fn, obj))
        else:
            return func(obj)
    #
    
    assert isinstance(lis, list), type(lis)
    
    return _nested_fn(lis)


