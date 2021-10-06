# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 17:08:46 2020

@author: TSAI, TUNG-CHEN
@update: 2021/10/06
@pipeline:
    1.
"""

SUBSET_IDS_PATH = r"config/subset_ids.json"


import os
import pkgutil
import numpy as np
from typing import Optional

from .lib.json.io import loads_json
from .preprocessors import DTYPE, search_files, parse_datapath, load_png

_raw = pkgutil.get_data(__package__, SUBSET_IDS_PATH)
SubsetIds = loads_json(_raw)
Keys = ['all', 'train', 'valid', 'test', 'fold0', 'fold1', 'fold2', 'fold3']
# =============================================================================
# ---- Functions
# =============================================================================
def get_subset_ids(subset: str = 'all') -> Optional[list]:
    if subset not in Keys:
        raise ValueError("`subset` should be any of %r " % Keys + 
                         "but got %r" % subset)
    if subset == 'all':
        return None
    
    if subset in {'train', 'valid', 'test'}:
        return SubsetIds[subset]
    idx = int(subset[-1])
    return SubsetIds['kfold'][idx]



def fetch(data: dict, subset: str = 'all') -> dict:
    assert isinstance(data, dict), type(data)
    subset_ids = get_subset_ids(subset)
    if subset_ids is None:
        return data
    
    in_subset = lambda p: any(map(
        lambda id_: id_ in os.path.basename(p), subset_ids))
    
    retained = list(map(in_subset, data['names']))
    for key in ['names', 'Ss', 'rpms', 'labels']:
        retained_gen = ( b for b in retained )
        data[key] = list(filter(lambda _: next(retained_gen), data[key]))
    
    return data


# =============================================================================
# ---- Classes
# =============================================================================
class DataCollector:
    def __init__(self, dtype=DTYPE, label_type='index', print_fn=print):
        self.dtype = dtype
        self.label_type = label_type
        self.print_fn = print_fn or (lambda *args, **kwargs: None)
    
    def __call__(self, directory, walk=False, is_SPL=True) -> dict:
        self.directory = directory
        self.walk = walk
        self.is_SPL = is_SPL
        
        png_paths = self.search_files(directory)
        if not png_paths:
            raise FileNotFoundError("No PNG files found in %r" % directory)
        
        data = self.read_files(png_paths)
        
        return data
    
    def search_files(self, directory) -> list:
        return search_files(
            directory, exts=['.png', '.PNG'], walk=self.walk, sort=True)
    
    def read_files(self, fpaths) -> dict:
        data_generator = map(self.read_file, fpaths)
        names, Ss, rpms, labels = self.unzip(data_generator)
        info = self.get_info(labels)
        
        return {
            "info": info, 
            "names": names, 
            "Ss": Ss, 
            "rpms": rpms, 
            "labels": labels
        }
    
    def read_file(self, fpath) -> tuple:
        S = load_png(fpath, is_SPL=self.is_SPL, dtype=self.dtype)
        name, rpm, label = parse_datapath(fpath, 
                                          label_type=self.label_type, 
                                          rpm_dtype=self.dtype)
        
        return name, S, rpm, label
    
    def unzip(self, iterable):
        data = np.array(list(iterable), dtype=object)
        return tuple( d.tolist() for d in data.T )
    
    def get_info(cls, labels):
        unique, counts = np.unique(labels, return_counts=True)
        n_classes = len(unique)
        n_total = counts.sum()
        class_counts = dict(zip(unique, counts))
        class_ratios = { c: n/n_total for c, n in class_counts.items() }
        class_weight = { c: (1/r)/n_classes for c, r in class_ratios.items() }
        
        return {    # data info
            "number of classes": n_classes, 
            "number of data": n_total, 
            "number of data in each class": class_counts, 
            "ratios of classes": class_ratios, 
            "class_weight": class_weight
        }



class SubsetDataCollector(DataCollector):
    def __call__(self, directory, walk=False, is_SPL=True, subset='all') -> dict:
        self.subset = subset.lower()
        return super().__call__(directory, walk=walk, is_SPL=is_SPL)
    
    def search_files(self, directory) -> list:
        fpaths = super().search_files(directory)
        subset_ids = get_subset_ids(self.subset)
        
        if subset_ids is None:
            return fpaths
        
        in_subset = lambda p: any(map(
            lambda id_: id_ in os.path.basename(p), subset_ids))
        
        return list(filter(in_subset, fpaths))


