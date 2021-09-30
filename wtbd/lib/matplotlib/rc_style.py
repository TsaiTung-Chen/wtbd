# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 09:15:49 2020

@author: TSAI, TUNG-CHEN
@update: 2021/09/19
"""
"""
USAGE:
    import matplotlib as mpl
    from rc_style import rc_style1
    mpl.rcParams.update(rc_style1)

DEFAULT: 
Z-order	 Artist
==================================================================
0        Images (AxesImage, FigureImage, BboxImage)
1        Patch, PatchCollection
2        Line2D, LineCollection (including minor ticks, grid lines)
2.01     Major ticks
3        Text (including axes labels and titles)
5        Legend

TUTORIALS: 
https://matplotlib.org/3.3.0/tutorials/introductory/customizing.html
"""

import functools
import matplotlib.pyplot as plt
# =============================================================================
# ---- Style ( default: `matplotlib.rcdefaults()` )
# =============================================================================
rc_style1 = {
    "text.color":'black', 
    
    "figure.figsize": (8, 4.5), 
    "figure.dpi": 120, 
#    "figure.autolayout": True, 
    "figure.constrained_layout.use": True, 
    "figure.facecolor":'white', 
    
    "axes.formatter.limits": (-2, 3), 
    "axes.formatter.use_mathtext": True,
    "axes.grid": True, 
    "axes.labelcolor": 'black', 
    "axes.facecolor": '#E6E6E6', 
    "axes.edgecolor": 'black', 
    "axes.linewidth": 0.7, 
    
    "grid.color": 'white', 
    
    "lines.linewidth": 0.7, 
    "lines.markersize": 4, 
    "lines.markeredgewidth": 1.2, 
    
    "legend.loc": 'best', 
    "legend.fontsize": 'small', 
    
    "xtick.color": 'black', 
    
    "ytick.color": 'black', 
    
    "pcolor.shading": 'auto'
}

rc_style2 = {
    "text.color":'black', 
    
    "figure.figsize": (8, 4.5), 
    "figure.dpi": 180, 
#    "figure.autolayout": True, 
    "figure.constrained_layout.use": True, 
    "figure.facecolor":'white', 
    
    "axes.formatter.limits": (-2, 3), 
    "axes.formatter.use_mathtext": True,
    "axes.grid": True, 
    "axes.labelcolor": 'black', 
    "axes.facecolor": '#E6E6E6', 
    "axes.edgecolor": 'white', 
    "axes.linewidth": 0.7, 
    
    "grid.color": 'white', 
    
    "lines.linewidth": 0.7, 
    "lines.markersize": 4, 
    "lines.markeredgewidth": 1.2, 
    
    "legend.loc": 'best', 
    "legend.fontsize": 'small', 
    
    "xtick.color": 'black', 
    
    "ytick.color": 'black', 
    
    "pcolor.shading": 'auto'
}


# =============================================================================
# ---- Utilities
# =============================================================================
def plt_rc_context(rc_style: dict = rc_style1):
    def _decorator(fn):
        @functools.wraps(fn)
        def __wrapper(*args, **kwargs):
            with plt.rc_context(rc_style):
                return fn(*args, **kwargs)
        return __wrapper
    return _decorator


