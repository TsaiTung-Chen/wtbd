# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:19:24 2020

@author: TSAI, TUNG-CHEN
@update: 2021/10/06
"""
"""
USAGE:
    from tictoc import TicToc
    tt = TicToc()
    
    tt.tic()
    # process...
    tt.toc()
"""


import time
# =============================================================================
# ---- Classes
# =============================================================================
class TicToc():
    def __init__(self, print_fn=print):
        self.print_fn = print_fn or (lambda *args, **kwargs: None)
        self.tics = []
    
    def tic(self, show=False, localtime=False):
        i = len(self.tics)
        self.tics.append(time.time())
        
        if show:
            self.print_fn('@ tic %d|---->\n' % i, end='' if localtime else '\n')
        if localtime:
            self.print_fn('@ Local time: %s\n' % self.localtime(show=False)[0])
        
        return 0
    
    def toc(self, show=False, format_on=True, pop=True, localtime=False):
        if len(self.tics)==0:
            self.print_fn('(None)\n')
            return None
        
        i = len(self.tics) - 1
        elapsed_sec = time.time() - self.tics.pop() if pop \
                 else time.time() - self.tics[-1]
        
        if show:
            self.print_fn('\n@ ---->| toc %d' % i)
            if format_on:
                days, remainder = divmod(elapsed_sec, 3600*24)
                hours, remainder = divmod(remainder, 3600)
                minutes, seconds = divmod(remainder, 60)
                self.print_fn('@ Elapsed time:', end=' ')
                self.print_fn(
                    '{} days {:02}:{:02}:{:02}\n'.format(
                        int(days), int(hours), int(minutes), int(seconds)), 
                    end='' if localtime else '\n'
                )
            else:
                self.print_fn('@ Elapsed time: %.6f seconds.\n' % elapsed_sec, 
                      end='' if localtime else '\n')
        
        if localtime:
            self.print_fn('@ Local time: %s\n' % self.localtime(show=False)[0])
        
        return elapsed_sec
    
    def localtime(self, show=False, colon=True):
        return localtime(show=show, colon=colon, print_fn=self.print_fn)


# =============================================================================
# ---- Functions
# =============================================================================
def localtime(show=False, colon=True, print_fn=print) -> (str, object):
    timeformat = "%Y-%m-%d_%H:%M:%S" if colon else "%Y-%m-%d_%H-%M-%S"
    localtime_obj = time.localtime()
    localtime_str = time.strftime(timeformat, localtime_obj)
    
    if show:
        print_fn('@ Local time:', localtime_str)
    
    return localtime_str, localtime_obj


# =============================================================================
#%% Test
# =============================================================================
if __name__ == '__main__':
    timer = TicToc(print)
    timer.tic(True, localtime=True)
    timer.toc(True, localtime=True)
    timer.localtime(True, colon=False)


