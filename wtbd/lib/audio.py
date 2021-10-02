# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 16:10:36 2020

@author: TSAI, TUNG-CHEN
@update: 2021/10/01
@inputs: audio.wav currently
"""

import librosa
import numpy as np
import audio_metadata
import soundfile as sf
from tabulate import tabulate

from .file import makedirs
# =============================================================================
# 
# =============================================================================
def get_streaminfo(fpath):
    return audio_metadata.load(fpath)['streaminfo']



def print_streaminfo(streaminfo, ncols=1, print_fn=print):
    streaminfo = str(streaminfo).split('\n', 1)[1].rsplit('\n', 1)[0]
    streaminfo = filter(None, streaminfo.split(' ' * 4))
    
    table = list()
    for info in streaminfo:
        key, colon, value = info.partition(': ')
        key = key.strip("'") + colon
        value = value.rsplit(',', 1)[0]
        value = value.strip("'")
        table.append((key, value))
    
    n_entries = int( np.ceil(len(table) / ncols) * ncols )
    n_trailings = n_entries - len(table)
    table += [('', '')] * n_trailings
    table = np.asarray(table, dtype=str).reshape(-1, ncols * 2)
    print_fn(tabulate(table, showindex=False, tablefmt='plain'))
    
    return table



def stereo_to_mono(load_path, 
                   save_path, 
                   channel, 
                   subtype='PCM_16', 
                   **kwargs):
    data, sr = librosa.load(load_path, sr=None, mono=False, dtype='float64')
    if data.ndim > 1:
        data = data[channel]
    makedirs(fpath=save_path)
    sf.write(save_path, data, sr, subtype=subtype, **kwargs)
    
    return data, sr


