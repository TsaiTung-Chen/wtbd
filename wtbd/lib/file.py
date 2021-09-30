# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:19:24 2020

@author: TSAI, TUNG-CHEN
@update: 2021/09/25
"""

import os
from datetime import datetime
# =============================================================================
### Functions
# =============================================================================
def insert_into_fname(fpath, string, at_end=True, makedir=False):
    filename, ext = os.path.splitext(os.path.basename(fpath))
    filename = '' if filename == '*' else filename
    
    folderpath = makedirs(fpath=fpath) if makedir else os.path.dirname(fpath)
    
    if at_end:
        return os.path.join(
            folderpath, 
            filename + string + ext
        )
    return os.path.join(
        folderpath, 
        string + filename + ext
    )


def insert_time_suffix(
        fpath, 
        at_end=True, 
        joint='_', 
        date=True, 
        time=True, 
        hyphen=True, 
        makedir=False):
    if date:
        if time:
            time_suffix = '{:%Y-%m-%d(%H-%M-%S)}'.format(datetime.now())
        else:
            time_suffix = '{:%Y-%m-%d}'.format(datetime.now())
    else:
        if time:
            time_suffix = '{:%H-%M-%S}'.format(datetime.now())
        else:
            raise ValueError("Either `date` or `time` should be True!")
    
    if not hyphen:
        time_suffix = time_suffix.replace('-', '')
    
    if joint:
        fname, ext = os.path.splitext(os.path.basename(fpath))
        if (fname != '*') or (joint != '_'):
            if at_end:
                time_suffix = joint + time_suffix
            else:
                time_suffix += joint
    
    return insert_into_fname(fpath, time_suffix, at_end=at_end, makedir=makedir)


def makedirs(fpath=None, folderpath=None):
    if folderpath is None:
        folderpath = os.path.dirname(fpath)
    
    if folderpath and (not os.path.isdir(folderpath)):
        os.makedirs(folderpath)
    
    return folderpath

