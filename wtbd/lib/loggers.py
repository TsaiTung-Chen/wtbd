# -*- coding: utf-8 -*-
"""
Created on Sun May  3 23:23:57 2020

@author: TSAI, TUNG-CHEN
@update: 2021/05/07

USAGE:
    loglevel = 'INFO'
    logpath = r"./logs/process_log.txt"
    from create_loggers import Loggers
    loggers = Loggers(loglevel, handlers=['stream', 'file'], logpath=logpath)
    
    loggers.sfinfo('message...')
"""

import os
import sys
import logging
import contextlib

from typing import Union
from typing_extensions import Literal

try:
    sys_path = list(sys.path)
    sys.path.insert(0, os.path.dirname(__file__))
    from tictoc import localtime
    from file import insert_time_suffix
finally:
    sys.path = sys_path
    del sys_path


Shutdown = Literal['shutdown']
Flush = Literal['flush']
Close = Literal['close']


class Loggers():
    def __init__(self, 
                 loglevel='WARNING', 
                 fmt='%(message)s', 
                 handlers=['stream'], 
                 logpath=r"./logs/log.txt", 
                 filemode='w'):
        valid_handlers = {'stream', 'file', 'dummy'}
        
        if isinstance(handlers, str):
            handlers = [handlers]
        if not set(handlers).issubset(valid_handlers):
            raise ValueError(
                "`handlers` should be a subset of {}.".format(valid_handlers))
        
        self.handlers = ['dummy', 'stream'] if 'dummy' in handlers else handlers
        self.config = {
            'loglevel': loglevel, 
            'fmt': fmt, 
            'logpath': logpath, 
            'filemode': filemode
        }
        
        self.list = []
        if 'stream' in handlers:
            self.snan = create_logger(r"stream_''", handlers='stream', 
                                      terminator='', **self.config)
            self.list.append(self.snan)
        
        if 'file' in handlers:
            self.fnan = create_logger(r"file_''", handlers='file', 
                                      terminator='', **self.config)
            self.list.append(self.fnan)
    
    def _log(self, print_fn, handler, msg='', *args, end='\n', **kwargs):
        if 'dummy' in self.handlers:
            return
        if handler in self.handlers:
            print_fn(msg, *args, **kwargs)
            print_fn(end)
    
    def sdebug(self, msg='', *args, end='\n', **kwargs):
        self._log(self.snan.debug, 'stream', msg, *args, end=end, **kwargs)
    
    def fdebug(self, msg='', *args, end='\n', **kwargs):
        self._log(self.fnan.debug, 'file', msg, *args, end=end, **kwargs)
    
    def sfdebug(self, msg='', *args, end='\n', **kwargs):
        self.sdebug(msg, *args, end=end, **kwargs)
        self.fdebug(msg, *args, end=end, **kwargs)
    
    def sinfo(self, msg='', *args, end='\n', **kwargs):
        self._log(self.snan.info, 'stream', msg, *args, end=end, **kwargs)
    
    def finfo(self, msg='', *args, end='\n', **kwargs):
        self._log(self.fnan.info, 'file', msg, *args, end=end, **kwargs)
    
    def sfinfo(self, msg='', *args, end='\n', **kwargs):
        self.sinfo(msg, *args, end=end, **kwargs)
        self.finfo(msg, *args, end=end, **kwargs)
    
    def swarning(self, msg='', *args, end='\n', **kwargs):
        self._log(self.snan.warning, 'stream', msg, *args, end=end, **kwargs)
    
    def fwarning(self, msg='', *args, end='\n', **kwargs):
        self._log(self.fnan.fwarning, 'file', msg, *args, end=end, **kwargs)
    
    def sfwarning(self, msg='', *args, end='\n', **kwargs):
        self.swarning(msg, *args, end=end, **kwargs)
        self.fwarning(msg, *args, end=end, **kwargs)
    
    def serror(self, msg='', *args, end='\n', **kwargs):
        self._log(self.snan.error, 'stream', msg, *args, end=end, **kwargs)
    
    def ferror(self, msg='', *args, end='\n', **kwargs):
        self._log(self.fnan.error, 'file', msg, *args, end=end, **kwargs)
    
    def sferror(self, msg='', *args, end='\n', **kwargs):
        self.serror(msg, *args, end=end, **kwargs)
        self.ferror(msg, *args, end=end, **kwargs)
    
    def scritical(self, msg='', *args, end='\n', **kwargs):
        self._log(self.snan.critical, 'stream', msg, *args, end=end, **kwargs)
    
    def fcritical(self, msg='', *args, end='\n', **kwargs):
        self._log(self.fnan.critical, 'file', msg, *args, end=end, **kwargs)
    
    def sfcritical(self, msg='', *args, end='\n', **kwargs):
        self.scritical(msg, *args, end=end, **kwargs)
        self.fcritical(msg, *args, end=end, **kwargs)
    
    def _execution_info(msg):
        def __decorator(fn):
            def ___wrapper(self, silent=False, empty_line=True, time=True):
                if not silent and ('dummy' not in self.handlers):
                    prefix = '\n' if empty_line else ''
                    string, obj = localtime(show=False)
                    prefix = prefix + string + ' ' if time else prefix
                    self.sfinfo(prefix + msg)
                return fn(self)
            return ___wrapper
        return __decorator
    
    @_execution_info('[Shutdown loggers]')
    def shutdown(self):
        """Informs the logging system to perform an orderly shutdown by 
           flushing and closing all handlers. 
           This should be called at application exit and no further use of the 
           logging system should be made after this call.
        """
        logging.shutdown()
    
    @_execution_info('[Flush loggers]')
    def flush(self):
        for my_logger in self.list:
            for my_handler in my_logger.handlers:
                my_handler.flush()
    
    @_execution_info('[Close loggers]')
    def close(self):
        for my_logger in self.list:
            for my_handler in my_logger.handlers:
                my_handler.close()
    
    def context(self, action: Union[Shutdown, Flush, Close] = 'shutdown'):
        return context(self, action=action)



def create_logger(name='', 
                  loglevel='WARNING', 
                  fmt='%(message)s', 
                  handlers=['stream'], 
                  terminator='\n', 
                  logpath=r"./logs/log.txt", 
                  filemode='w'):
    valid_handlers = {'stream', 'file'}
    
    if isinstance(handlers, str):
        handlers = [handlers]
    if not set(handlers).issubset(valid_handlers):
        raise ValueError(
            "`handlers` should be a subset of {}.".format(valid_handlers))
    
    # Config
    formatter = logging.Formatter(fmt=fmt, datefmt="%H:%M:%S")
    my_logger = logging.getLogger(name)
    my_logger.setLevel(loglevel)
    my_logger.handlers.clear()
    
    if 'file' in handlers:    # Log file
        logpath = insert_time_suffix(logpath, hyphen=False, makedir=True)
        
        # File Handler
        fileHandler = logging.FileHandler(logpath, filemode, 'utf-8')
        fileHandler.setFormatter(formatter)
        fileHandler.setLevel(loglevel)
        fileHandler.terminator = terminator
        my_logger.addHandler(fileHandler)
    
    if 'stream' in handlers:    # Stream Handler
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(formatter)
        consoleHandler.setLevel(loglevel)
        consoleHandler.terminator = terminator
        my_logger.addHandler(consoleHandler)
    
    return my_logger



@contextlib.contextmanager
def context(
        loggers: Loggers, action: Union[Shutdown, Flush, Close] = 'shutdown'):
    action = action.lower()
    
    try:
        yield loggers
    finally:
        if action == 'shutdown':
            loggers.shutdown()
        elif action == 'flush':
            loggers.flush()
        elif action == 'close':
            loggers.close()


