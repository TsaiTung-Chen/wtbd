#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 18:21:32 2021

@author: TSAI, TUNG-CHEN
@update: 2021/10/05
"""

import functools
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from typing import Optional, Iterable, Callable

from .preprocessors import change_symbol
from .preprocessors import VLIM_PA, VLIM_SPL
from .lib.matplotlib.rc_style import plt_rc_context
# =============================================================================
# 
# =============================================================================
def get_inputs(modelname: str, data: dict) -> dict:
    if 'simple' in modelname.lower():
        return {"input_sp": data['Ss']}
    return {"input_sp": np.asarray(data['Ss']), 
            "input_rs": np.asarray(data['rpms'])}



def get_targets(data: dict):
    return np.asarray(data['labels'])



@plt_rc_context()
def plot_spectrogram(S, 
                     tlimits: Optional[Iterable] = None, 
                     flimits: Optional[Iterable] = None, 
                     is_SPL=True, 
                     title=None):
    S = np.asarray(S)
    assert S.ndim == 2, S.ndim
    
    vmin, vmax = VLIM_SPL if is_SPL else VLIM_PA
    clabel = 'dB SPL' if is_SPL else 'pressure amplitude [Pa]'
    
    fig, ax = plt.subplots()
    plt.suptitle(title)
    
    if (flimits is None) or (tlimits is None):
        plt.pcolormesh(S, vmin=vmin, vmax=vmax)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
    else:
        F = np.linspace(*flimits, S.shape[0])
        T = np.linspace(*tlimits, S.shape[1])
        plt.pcolormesh(T, F, S, vmin=vmin, vmax=vmax)
        plt.xlabel('time [sec]')
        plt.ylabel('fequency [Hz]')
    plt.colorbar(label=clabel)
    plt.show()
    
    return fig



def print_info(dictionary: dict,  print_fn: Optional[Callable] = print):
    def str_(value):
        if np.issubdtype(type(value), np.floating):
            return '{:.4f}'.format(value)
        return str(value)
    
    #
    string = ''
    for key, value in dictionary.items():
        string += '%s: ' % key
        if isinstance(value, dict):
            substr = [ '%s = %s' % (k, str_(v)) for k, v in value.items() ]
            substr = ', '.join(substr)
            string += '{' + substr + '}'
        elif isinstance(value, set):
            string += '{' + ', '.join([str_(v) for v in value]) + '}'
        elif isinstance(value, tuple):
            string += '(' + ', '.join([str_(v) for v in value]) + ')'
        elif isinstance(value, list):
            string += '[' + ', '.join([str_(v) for v in value]) + ']'
        else:
            string += str_(value)
        string += '\n'
    
    if print_fn:
        print_fn(string)
    
    return string



def print_prediction(data: dict, pred, label_type='name', print_fn=print):
    names, pred = np.squeeze(data['names']), np.squeeze(pred)
    assert len(names) == len(pred)
    
    _change_symbol = functools.partial(change_symbol, symbol_type=label_type)
    pred = list(map(_change_symbol, pred))
    index = range(len(names))
    table = tabulate(np.stack([index, names, pred], axis=-1), 
                     headers=['index', 'name', 'predicted class'])
    print_fn(table)
    
    return table



def print_metrics(results: dict, print_fn=print):
    string = [ '%s: %.4f' % (k, v) for k, v in results.items() ]
    string = ' - '.join(string)
    print_fn(string)
    
    return string


