#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 18:21:32 2021

@author: TSAI, TUNG-CHEN
@update: 2021/10/04
"""

MODEL_DIRECTORY = r"models/"


import os
import sys
import contextlib
import numpy as np
import tensorflow as tf
from typing import Optional
from tabulate import tabulate

from .postprocess import binary_classification

_pkg_path = os.path.dirname(sys.modules[__package__].__file__)
ModelDirectory = os.path.join(_pkg_path, MODEL_DIRECTORY)

ModelNames = {'SimpleCNN', 'PhysicalNN', 'PhysicalCNN', 'PhysicalCNNCI'}
ModelPaths = { n.lower(): os.path.join(ModelDirectory, n) for n in ModelNames }
# =============================================================================
# ---- Functions
# =============================================================================
@contextlib.contextmanager
def check_numerics():
    try:
        tf.debugging.enable_check_numerics()
        yield
    finally:
        tf.debugging.disable_check_numerics()



def plot_tensor_flow(model, to_file: Optional[str] = None, expand_nested=True):
    if to_file is None:
        return tf.keras.utils.plot_model(model, 
                                         show_shapes=True, 
                                         show_layer_names=True, 
                                         expand_nested=expand_nested)
    else:
        folderPath = os.path.dirname(os.path.abspath(to_file))
        
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        
        return tf.keras.utils.plot_model(model, 
                                         to_file, 
                                         show_shapes=True, 
                                         show_layer_names=True, 
                                         expand_nested=expand_nested)



def print_model_tree(model, 
                     classname: bool = True, 
                     backend=False, 
                     print_fn=print, 
                     depth=0):
    prefix = f'{depth}|' + '―' * (2*depth - 1)
    
    print_fn(prefix, end=' ')
    
    name = getattr(model, 'name', '[NO NAME]')
    if classname:
        print_fn('%s (%s)' % (name, model.__class__.__qualname__))
    else:
        print_fn(name)
    
    if backend:
        layers = getattr(model, '_layers', None)
    else:
        layers = getattr(model, 'layers', None)
    if layers is not None:
        for layer in layers:
            print_model_tree(layer, 
                             classname=classname, 
                             backend=backend, 
                             print_fn=print_fn, 
                             depth=depth + 1)



def print_trainable(model, classname: bool = False, print_fn=print):
    def _get_info(model, depth=0):
        index = f'{depth}'
        
        prefix = '―' * (2*depth - 1)
        
        name = getattr(model, 'name', '[NO NAME]')
        if classname:
            name = ' '.join([
                prefix, name, '(%s)' % model.__class__.__qualname__])
        else:
            name = ' '.join([prefix, name])
        
        trainable = model.trainable
        
        n_trainable = sum([ 
            tf.size(t).numpy() for t in model.trainable_weights ])
        n_nontrainable = sum([ 
            tf.size(t).numpy() for t in model.non_trainable_weights ])
        if n_trainable == 0:
            n_trainable = None
        if n_nontrainable == 0:
            n_nontrainable = None
        
        table.append([index, name, trainable, n_trainable, n_nontrainable])
        
        if hasattr(model, 'layers'):
            for layer in model.layers:
                _get_info(layer, depth=depth + 1)
    ##
    
    columns = ['name', 'trainable', 'n_train', 'n_nontrain']
    table = list()
    _get_info(model, depth=0)
    table = np.asarray(table)
    
    print_fn(tabulate(table, 
                      headers=columns, 
                      tablefmt="fancy_grid", 
                      showindex=False, 
                      numalign='right'))



def get_layer(model, name_or_index):
    def _get_layer(model):
        try:
            return model.get_layer(name=name, index=index)
        except ValueError:
            if hasattr(model, 'layers'):
                for layer in model.layers:
                    if not hasattr(layer, 'get_layer'):
                        continue
                    layer = _get_layer(layer)
                    if layer is not None:
                        return layer
    ##
    
    if model.name == name_or_index:
        return model
    
    if isinstance(name_or_index, str):
        name = name_or_index
        index = None
    else:
        name = None
        index = name_or_index
    
    got_layer = _get_layer(model)
    
    if got_layer:
        return got_layer
    raise ValueError("Layer '%s' not found." % name_or_index)



def switch_trainable(model, 
                     names_or_indices, 
                     to: bool, 
                     show=True, 
                     classname: bool = True, 
                     print_fn=print):
    assert isinstance(to, bool)
    
    if names_or_indices is None:
        return
    elif not isinstance(names_or_indices, (list, tuple)):
        names_or_indices = [names_or_indices]
    
    for name_or_index in names_or_indices:
        layer = get_layer(model, name_or_index)
        layer.trainable = to
        
        if not show:
            continue
        if classname:
            print_fn("layer: '%s' (%s) trainable=%s" % (
                layer.name, 
                layer.__class__.__qualname__, 
                layer.trainable))
            continue
        print_fn("layer: '%s' trainable=%s" % (
            layer.name, 
            layer.trainable))



def get_network(modelname, *args, **kwargs):
    path = ModelPaths.get(modelname.lower(), None)
    if path is None:
        raise ValueError("`modelname` should be any of %r but got %r" % (
            ModelNames, modelname))
    
    model = tf.keras.models.load_model(path, *args, **kwargs)
    
    return Network(model, modelname)


# =============================================================================
# ---- Classes
# =============================================================================
class Network:    # Wrapper
    def __init__(self, model, modelname):
        self.model = model
        self.modelname = modelname
        self.__call__ = model.__call__
    
    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, **kwargs):
        return self.model.evaluate(x, 
                                   y, 
                                   batch_size=batch_size, 
                                   verbose=verbose, 
                                   return_dict=True, 
                                   **kwargs)
    
    def predict(self, x, batch_size=None, verbose=1, **kwargs):
        return self.model.predict(
            x, batch_size=batch_size, verbose=verbose, **kwargs)
    
    def infer(self, x, batch_size=None, verbose=1, **kwargs):
        binary_accuracy = self.get_metric('accuracy')
        threshold = binary_accuracy.get_config()['threshold']
        outputs = self.predict(
            x, batch_size=batch_size, verbose=verbose, **kwargs)
        
        return binary_classification(outputs, threshold=threshold)
    
    def get_metric(self, name=None, index=None):
        if name is not None:
            for metric in self.metrics:
                if metric.name == name:
                    return metric
        if index is not None:
            return self.metrics[index]
        raise ValueError("Metric with `name` %r or `index` %r not found." % (
            name, index))
    
    def save(self, filepath, *args, **kwargs):
        return self.model.save(filepath, *args, **kwargs)
    
    def plot_tensor_flow(self, to_file=None, expand_nested=True):
        return plot_tensor_flow(self.model, 
                                to_file=to_file, 
                                expand_nested=expand_nested)
    
    def summary(self, *args, **kwargs):
        return self.model.summary(*args, **kwargs)
    
    def get_layer(self, name_or_index):
        return get_layer(model, name_or_index)
    
    @property
    def layers(self):
        return self.model.layers
    
    @property
    def optimizer(self):
        return self.model.optimizer
    
    @property
    def metrics(self):
        return self.model.metrics
    
    @property
    def loss(self):
        return self.model.loss


