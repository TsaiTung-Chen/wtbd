# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 17:08:46 2020

@author: TSAI, TUNG-CHEN
@update: 2021/09/30
@pipeline:
    1.
"""

SUBSET_IDS_PATH = r"config/subset_ids.json"


import pkgutil
import numpy as np
from typing import Optional, Callable

from .preprocessors import DTYPE, search_files, parse_datapath, load_png
from .lib.json.io import loads_json

_raw = pkgutil.get_data(__package__, SUBSET_IDS_PATH)
SubsetIds = loads_json(_raw)
# =============================================================================
# ---- Functions
# =============================================================================
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
        print_info(data['info'], print_fn=self.print_fn)
        
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
        name, rpm, label = parse_datapath(fpath, label_type=self.label_type)
        
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
    SubsetIds = SubsetIds
    Keys = {'all', 'train', 'valid', 'test', 'fold0', 'fold1', 'fold2', 'fold3'}
    
    def __call__(self, directory, subset, walk=False, is_SPL=True) -> dict:
        self.subset = subset.lower()
        return super().__call__(directory, walk=walk, is_SPL=is_SPL)
    
    def search_files(self, directory) -> list:
        if self.subset not in self.Keys:
            raise ValueError("`subset` should be any of %r " % self.Keys + 
                             "but got %r" % self.subset)
        
        fpaths = super().search_files(directory)
        if self.subset == 'all':
            return fpaths
        
        if self.subset in {'train', 'valid', 'test'}:
            ids = self.SubsetIds[self.subset]
        else:
            idx = int(self.subset[-1])
            ids = self.SubsetIds['kfold'][idx]
        
        in_subset = lambda p: any(map(lambda id_: id_ in p, ids))
        
        return list(filter(in_subset, fpaths))


