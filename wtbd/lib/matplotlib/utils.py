# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 00:58:48 2021

@author: TSAI, TUNG-CHEN
@update: 2021/09/25
"""

import os
from ..file import insert_time_suffix, makedirs
# =============================================================================
# 
# =============================================================================
def save_fig(fig, fpath, time_suffix=False, newfolder=False, makedir=True):
    if newfolder:
        folderpath = os.path.dirname(fpath)
        fname = os.path.basename(fpath)
        folderpath = os.path.join(folderpath, 'plots')
        fpath = os.path.join(folderpath, fname)
    
    if time_suffix:
        fpath = insert_time_suffix(fpath, hyphen=False)
    if makedir:
        makedirs(fpath)
    
    fname, ext = os.path.splitext(fpath)
    if ext == '':
        fpath += '.png'
    
    fig.savefig(fpath, dpi='figure')


