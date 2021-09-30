#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 18:21:32 2021

@author: TSAI, TUNG-CHEN
@update: 2021/09/28
"""

import numpy as np
import tensorflow as tf
from typing import Optional, Iterable
import matplotlib.pyplot as plt

from .preprocessors import DTYPE, VLIM_PA, VLIM_SPL
from .lib.matplotlib.rc_style import plt_rc_context
# =============================================================================
# 
# =============================================================================
def convert_to_tensor(iterable, dtype=DTYPE):
    shapes = np.asarray(list(map(lambda arr: np.shape(arr), iterable)))
    same_shape = (shapes == shapes[0]).all()
    
    if same_shape:
        return tf.convert_to_tensor(iterable, dtype=dtype)
    return tf.ragged.constant(iterable, dtype=dtype)



def get_inputs(modelname: str, data: dict) -> dict:
    if 'simple' in modelname.lower():
        return {"input_sp": convert_to_tensor(data['Ss'])}
    return {"input_sp": convert_to_tensor(data['Ss']), 
            "input_rs": convert_to_tensor(data['rpms'])}



def get_targets(data: dict):
    return convert_to_tensor(data['labels'])



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


