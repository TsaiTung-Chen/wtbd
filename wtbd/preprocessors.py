# -*- coding: utf-8 -*-
"""
Created on Sat eb  8 21:52:55 2020

@author: TSAI, TUNG-CHEN
@update: 2021/10/04
@outputs: numpy array in float64
@pipeline:
    1.
"""

P_REF = 20e-6    # ref pressure: 20 (muPa)
CHANNEL = 0    # Channel index
DTYPE = 'float32'

# Value range
VLIM_SPL = (-10, 100)    # dB SPL
VLIM_PA = (0, 0.0005)    # acoustic pressure in Pa <=> (-inf~28 dB SPL)
EPSILON = 1e-7    # to avoid ZeroDivisionError

# Classes
INDEX_TO_NAME = { 0: "[n]", 1: "[d]" }
NAME_TO_INDEX = {
    "[n]": 0, "[d]": 1, "[0d]": 0, "[1d]": 1, "[2d]": 1, "[3d]": 1}

RECORDING_ENVIRONMENTS_PATH = r"config/recording_environments.json"


import os
import re
import shutil
import pkgutil
import librosa
import functools
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import rfftfreq, rfft, fftfreq, fft, ifft
from tabulate import tabulate
import skimage.io as skimage_io
import skimage.exposure as skimage_exposure
import skimage.transform as skimage_transform
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from typing import List, Dict, Union

from .lib.json.io import loads_json
from .lib.matplotlib.utils import save_fig
from .lib.matplotlib.rc_style import plt_rc_context
from .lib.audio import get_streaminfo, print_streaminfo

_raw = pkgutil.get_data(__package__, RECORDING_ENVIRONMENTS_PATH)
RecordingEnvironments = loads_json(_raw)
RotorSpeeds = (9, 19)    # min and max rotor speeds (rpm)
# =============================================================================
# ---- Functions
# =============================================================================
def moving_average(x, n_points):
    """`n_points` moving average.
    """
    if isinstance(x, list) or isinstance(x, tuple):
        x = np.asarray(x)
    elif not isinstance(x, np.ndarray):
        raise TypeError("Invalid `x`!")
    
    kernel = np.ones(n_points, dtype=x.dtype) / n_points
    
    return np.convolve(x, kernel, mode='same')



def get_id(filepath) -> str:
    filename = os.path.basename(filepath)
    return os.path.splitext(filename)[0]



def get_record_info(filename) -> dict:
    patterns = {"date": r"@\d{8}", 
                "no": r"#\d+", 
                "classname": r"\[[^\[\]]+\]", 
                "subtitle": r"-\d+", 
                "period": r"\([^()]+s\)", 
                "rpm": r"\`[^`]+rpm\`"}
    record_info = dict.fromkeys(patterns.keys(), '')
    for key, pattern in patterns.items():
        info = re.findall(pattern, filename)
        if info:
            record_info[key] = info[0]
    
    return record_info



def join_record_info(info:Union[List[str], Dict[str, str]]) -> str:
    if isinstance(info, dict):
        info = [ info.get(k, '') for k in ("date", 
                                           "no", 
                                           "classname", 
                                           "subtitle", 
                                           "period", 
                                           "rpm") ]
    
    return '_'.join(filter(None, info))



def get_record_env(id_: str) -> dict:
    return RecordingEnvironments.get(id_, None)



def change_symbol(name_or_index, symbol_type='index'):
    assert symbol_type in {'index', 'name'}, symbol_type
    
    if isinstance(name_or_index, str):    # got name
        if symbol_type == 'index':
            return NAME_TO_INDEX[name_or_index]
        return INDEX_TO_NAME[NAME_TO_INDEX[name_or_index]]
    # got index
    if symbol_type == 'name':
        return INDEX_TO_NAME[name_or_index]
    return name_or_index



def generate_labels(name_or_index, label_type='index', n_samples=1):
    label = change_symbol(name_or_index, symbol_type='index')
    
    if n_samples == 1:
        return label
    return [label] * n_samples



def count_classes(name_or_index, n_samples=1):
    classname = change_symbol(name_or_index, symbol_type='name')
    keys = INDEX_TO_NAME.values()
    classcounts = dict.fromkeys(keys, 0)
    classcounts.update({classname: n_samples})
    
    return classcounts


    
def search_files(directory,
                 exts=['.wav'],
                 walk=False,
                 skip_hidden=True,
                 sort=False):
    if isinstance(exts, str):
        exts = [exts]
    
    if not os.path.isdir(directory) and any(map(directory.endswith, exts)):
        if skip_hidden and directory.startswith('.'):
            return list()
        return [directory]
    
    if skip_hidden:
        func = lambda f: f.endswith(ext) and (not f.startswith('.'))
    else:
        func = lambda f: f.endswith(ext)
    fpaths = list()
    for ext in exts:
        for direc, folders, files in os.walk(directory):
            files = filter(func, files)
            fpaths += list(map(lambda f: os.path.join(direc, f), files))
            if not walk:
                break
    
    return sorted(fpaths) if sort else fpaths



def parse_datapath(fpath, label_type='index', rpm_dtype=DTYPE) -> dict:
    id_ = get_id(fpath)
    info = get_record_info(id_)
    classname = info['classname']
    label = change_symbol(classname, symbol_type=label_type)
    
    rpm = info['rpm'][1:-4]
    rpm = np.dtype(rpm_dtype).type(rpm)
    
    return id_, rpm, label



def _array_mapping(f):    # decorator
    @functools.wraps(f)
    def decorated_f(X, *args, **kwargs):
        if isinstance(X, list):    # a list of ndarrays
            return list(map(lambda x: f(x, *args, **kwargs), X))
        if isinstance(X, np.ndarray):    # a ndarray
            return f(X, *args, **kwargs)
        raise TypeError("`X` should be an array or a list containing "
                        "arrays but got %s" % type(X))
    
    return decorated_f



@_array_mapping
def cast(X, dtype=DTYPE):
    if hasattr(X, 'astype'):
        return X.astype(dtype)
    return np.dtype(dtype).type(X)



@_array_mapping
def frequency_filter(S, 
                     F=None, 
                     flimits=[0, 12800], 
                     cutin_freq=None, 
                     cutoff_freq=None):
    assert S.ndim == 2, S.ndim
    
    if F is None:
        F = np.linspace(*flimits, S.shape[0])
    
    if (cutin_freq is None) and (cutoff_freq is None):
        return S, F
    
    cutin_freq = F[0] if cutin_freq is None else cutin_freq
    cutoff_freq = F[-1] if cutoff_freq is None else cutoff_freq
    
    mask = (cutin_freq <= F) & (F <= cutoff_freq)
    
    return S[mask], F[mask]



@_array_mapping
def to_SPL(X: Union[List[np.ndarray], np.ndarray], 
           minimum=VLIM_SPL[0], 
           epsilon=EPSILON):
    dtype = X.dtype if isinstance(X, np.ndarray) else type(X)
    p_rms = X / np.sqrt(2)    # amplitude => rms
    
    return np.maximum( 
        20*np.log10( p_rms/P_REF + epsilon ), 
        minimum, 
        dtype=dtype
    )



@_array_mapping
def to_Pa(X: Union[List[np.ndarray], np.ndarray], epsilon=EPSILON):
    dtype = X.dtype if isinstance(X, np.ndarray) else type(X)
    p_rms = ( 10 ** (X / 20) - epsilon ) * P_REF
    
    return (p_rms * np.sqrt(2)).astype(dtype)



@_array_mapping
def resize(S, shape=[None, None]):
    """`shape`: Iterable containing ints or None. The None dimensions in 
    `shape` will be remained.
    """
    if shape is None:
        return S
    
    assert S.ndim == len(shape) == 2, (S.ndim, len(shape))
    
    if None in shape:
        shape = tuple( (d2 or d1) for d1, d2 in zip(S.shape, shape) )
    
    if tuple(shape) == tuple(S.shape):
        return S
    return skimage_transform.resize(  # bi-linear interpolation
        S, shape, order=1, anti_aliasing=True)



def save_png(fpath, S, is_SPL=True, flip_freq_axis=True):
    """`S` has dtype 'float32' or 'float64' and will be rescale to [0, 1]. 
    After rescaling, map `S` to [0, 2**16-1] with dtype 'uint16'.
    """
    assert S.ndim == 2, S.ndim
    
    fdir = os.path.dirname(fpath)
    if (not os.path.exists(fdir)) and fdir:
        os.makedirs(fdir)
    if not fpath.endswith('.png'):
        fpath += '.png'
    
    vlim = VLIM_SPL if is_SPL else VLIM_PA
    S = S[::-1] if flip_freq_axis else S
    S = skimage_exposure.rescale_intensity(S, in_range=vlim, out_range=(0, 1))
    S *= (2**16 - 1)
    S = S.astype('uint16')
    skimage_io.imsave(fpath, S)



def load_png(fpath, 
             is_SPL=True, 
             flip_freq_axis=True, 
             dtype=DTYPE) -> np.ndarray:
    vlim = VLIM_SPL if is_SPL else VLIM_PA
    S = skimage_io.imread(fpath)
    assert S.dtype == 'uint16', S.dtype
    
    S = S[::-1] if flip_freq_axis else S
    S = S.astype('float64')
    S /= (2**16 - 1)
    
    return skimage_exposure.rescale_intensity(
        S, in_range=(0, 1), out_range=vlim).astype(dtype)



def is_empty(directory) -> bool:
    if os.path.isdir(directory) and os.listdir(directory):
        return False
    return True



def check_for_remove(directory, exceptions=[]):
    def remove():
        response = ''
        while response not in {'y', 'n'}:
            response = input('"{}" already exists. '.format(directory) + 
                             'Remove it? [y/n]: ').lower()
        return response == 'y'
    
    def find_fpaths(direc, files_or_folders):
        removed_paths = list()
        for f in files_or_folders:
            fpath = os.path.join(direc, f)
            if not any(map(lambda ex: fpath.startswith(ex), exceptions)):
                removed_paths.append(fpath)
        return removed_paths
    
    #
    if is_empty(directory):
        return
    
    if not exceptions:
        if remove():
            shutil.rmtree(directory)
        return
    
    if not isinstance(exceptions, (list, tuple)):
        exceptions = [exceptions]
    exceptions = list(map(os.path.abspath, exceptions))
    directory = os.path.abspath(directory)
    
    removed_fpaths = list()
    removed_folders = list()
    for direc, folders, files in os.walk(directory):
        removed_fpaths += find_fpaths(direc, files)
        removed_folders += find_fpaths(direc, folders)
    
    if (removed_fpaths or removed_folders) and remove():
        list(map(os.remove, removed_fpaths))
        list(map(shutil.rmtree, removed_folders))


# =============================================================================
# ---- Classes
# =============================================================================
class Preprocessor:
    def __init__(self, 
                 mode='dynamic',     # 'dynamic': RS-based; 'static': 20 sec
                 label_type='index', 
                 SPL=True, 
                 shape=[96, 96], 
                 dtype=DTYPE, 
                 begin=30,     # (sec) skip the first 30 sec
                 cutin_freq=4000,     # (Hz)
                 cutoff_freq=None,     # (Hz)
                 plot=3, 
                 render=True,     # render the plots
                 print_fn=print):
        assert mode in {'static', 'dynamic', 'dynamic0', 'dynamic012'}, mode
        assert plot in {0, 1, 2, 3}, plot
        self.mode = mode
        self.label_type = label_type
        self.SPL = SPL
        self.shape = shape
        self.dtype = dtype
        self.begin = begin
        self.cutin_freq = cutin_freq
        self.cutoff_freq = cutoff_freq
        self.plot = plot
        self.render = render
        self.print_fn = print_fn or (lambda *args, **kwargs: None)
        self.vlim = VLIM_SPL if SPL else VLIM_PA
    
    @plt_rc_context()
    def __call__(self, directory, walk=False, save_directory=None) -> dict:
         # keys: 'names', 'Ss', 'rpms', 'labels', 'info'
        self.directory = directory
        self.walk = walk
        self.save_directory = save_directory
        self.save = save_directory is not None
        self._idx_gen = Generator()
        
        wav_paths = search_files(directory, 
                                 exts=['.wav', '.WAV'], 
                                 walk=walk, 
                                 sort=True)
        n_wavs = str(len(wav_paths))
        progress = r"#%0" + str(len(n_wavs)) + r"d/%s"
        if int(n_wavs) == 0:
            raise FileNotFoundError("No WAV files found in %r" % directory)
        
        if self.save:
            self.print_fn('Saving directory: %s\n' % save_directory)
        
        results = dict()
        for i, filepath in enumerate(wav_paths, start=1):
            self.print_fn(progress % (i, n_wavs) + ' WAV file: %s' % filepath)
            
            # WAV file's metadata
            streaminfo = get_streaminfo(filepath)
            print_streaminfo(streaminfo, ncols=3, print_fn=self.print_fn)
            
            # Process
            res = self.process(filepath)
            
            # Resize
            if self.shape:
                res.update({"Ss": resize(res['Ss'], self.shape)})
            
            # Concat the results
            if i == 1:
                info = res.pop('classcounts')
                if 'table' in res.keys():
                    results['table'] = [res.pop('table')]
            else:
                info += res.pop('classcounts')
                if 'table' in res.keys():
                    results['table'].append(res.pop('table'))
            
            for key, value in res.items():
                results.setdefault(key, list()).extend(value)
            
            self.print_fn('')
            
            # Save
            if not self.save:
                continue
            join_path = functools.partial(os.path.join, save_directory, 'data')
            save_spectrogram = functools.partial(save_png, is_SPL=self.SPL)
            datapaths = map(join_path, res['names'])
            list(map(save_spectrogram, datapaths, res['Ss']))
        
        info['total'] = info.sum()
        if 'table' in results.keys():
            results['table'] = pd.concat(results['table'], ignore_index=True)
        
        # Write preprocess info
        if self.save:
            fpath = os.path.join(save_directory, 'preprocess_info.txt')
            with open(fpath, 'w') as f:
                print(info.to_string(), file=f)
                if 'table' in results.keys():
                    print(file=f)
                    print(
                        tabulate(results['table'], 
                                 headers='keys', 
                                 floatfmt='.4f', 
                                 colalign=('right', 'left', 'right', 'right')), 
                        file=f
                    )
        
        self.print_fn('Done!')
        results['info'] = info
        
        return results
    
    def process(self, filepath):
        if self.save and not (os.path.exists(self.save_directory)):
            os.makedirs(self.save_directory)
        
        # Load Wav File
        x, sr = librosa.load(filepath, sr=None, mono=False, dtype='float64')
         # ==> double(float64) normalized into -1.0 and +1.0
        
        # Search info
        id_ = get_id(filepath)
        record_info = get_record_info(id_)
        classname = record_info['classname']
        
        record_env = get_record_env(id_)
        if record_env:
            sensitivity = record_env['sensitivity']
            scale = record_env['scale']
        else:
            sensitivity = float(input("Enter the sensitivity (V/Pa): "))
            scale = float(input("Enter the scaling factor (usually 1): "))
        
        # Scale
        if x.ndim > 1:
            x = x[CHANNEL, :]
        x = x[self.begin*sr:]    # start from 30 sec
        x *= scale
        x /= sensitivity    # => (Pa)
        
        # Process
        if 'static' == self.mode:
            results = self.static_process(x, sr, 
                                          id_=id_, 
                                          scale=scale, 
                                          sensitivity=sensitivity)
        else:
            results = self.dynamic_process(x, sr, 
                                           id_=id_, 
                                           scale=scale, 
                                           sensitivity=sensitivity)
        
        n_samples = len(results['names'])
        results.update({
            "labels": generate_labels(classname,
                                      label_type=self.label_type, 
                                      n_samples=n_samples), 
            "classcounts": pd.Series(count_classes(classname, 
                                                   n_samples=n_samples))
        })
        
        if 'static' == self.mode:
            return results
        
        # Process info
        table = {
            "ID": id_, 
            "# of samples": n_samples, 
            "Averaged rotor speed (%s~%s)" % RotorSpeeds: 
                '%.2f rpm' % results.pop('avg_rpm'), 
            "Overload ratio": results.pop('overload')['ratio']
        }
        results['table'] = pd.DataFrame(table, index=[0])
        
        return results
    
    def static_process(self, x, sr, id_, scale, sensitivity) -> dict:
        clip_duration = 20    # (sec)
        clip_hop = clip_duration // 2    # (sec)
        n_clips = int(np.floor(1 + (len(x)/sr - clip_duration) / clip_hop))
         # number of 20-sec subdataset (with 50% overlap)
        
        Ss, names = list(), list()
        for c in range(n_clips):
            t1 = clip_hop * c
            t2 = t1 + clip_duration
            x_clip = x[int(t1*sr):int(t2*sr)]        # (point)
            t_start, t_stop = t1 + self.begin, t2 + self.begin
            
            # PNG name
            name = self.make_pngname(x, sr, id_, t_start, t_stop)
            S = self.static_process_clip(x_clip, 
                                         sr, 
                                         t_start=t_start, 
                                         t_stop=t_stop, 
                                         id_=id_)
            Ss.append(S)
            names.append(name)
        
        return {"Ss": Ss, "names": names}
    
    def static_process_clip(self, x, sr, t_start, t_stop, id_) -> tuple:
        # Spectrogram
        S, F, T, pps = self.spectrogram(x, sr, t_start=t_start)
        
        # Transformation
        S, F = frequency_filter(S, F=F, 
                                cutin_freq=self.cutin_freq, 
                                cutoff_freq=self.cutoff_freq)
        
        if self.SPL:
            S = to_SPL(S)
        
        S = cast(S, self.dtype)
        
        if self._plot < 1:
            return S
        
        # Plot
        clabel = 'dB SPL' if self.SPL else 'pressure amplitude [Pa]'
        
        fig, ax = plt.subplots()
        plt.suptitle(id_)
        plt.title('Spectrogram')
        plt.pcolormesh(T, F, S, vmin=self.vlim[0], vmax=self.vlim[1])
        plt.xlim(t_start, t_stop)
        plt.xlabel('time [sec]')
        plt.ylabel('fequency [Hz]')
        plt.colorbar(label=clabel)
        
        if self.render:
            plt.show()
        else:
            plt.close()
        if self.save:
            fpath = os.path.join(self.save_directory, 
                                 id_+'_%d.png' % next(self._idx_gen))
            save_fig(fig, fpath, newfolder=True, makedir=True)
        
        return S
    
    def dynamic_process(self, 
                        x, 
                        sr, 
                        id_, 
                        scale, 
                        sensitivity) -> dict:
        t_start = self.begin
        t_stop = len(x) / sr
        
        # Spectrogram
        S, F, T, pps = self.spectrogram(x, sr, t_start=t_start)
        
        # Detect overload
        overload = self.detect_overload(S, F, T, pps, 
                                        sensitivity=sensitivity, 
                                        scale=scale)
        
        # Estimate rotor speed
        bounds, irrationals, irrational_ratio = self.estimate_rotorspeed(
            S, F, T, pps, 
            id_=id_, 
            overload=overload
        )
        
        # Transformation
        S, F = frequency_filter(S, F=F, 
                                cutin_freq=self.cutin_freq, 
                                cutoff_freq=self.cutoff_freq)
        
        if self.SPL:
            S = to_SPL(S)
        
        # Split the spectrogram
        bounds = self.filter_bounds(bounds, overload, irrationals)
        S_split, T_split, rpms, avg_rpm = self.split_spectrogram(
            S, T, 
            bounds=bounds, 
            irrational_ratio=irrational_ratio
        )
        
        S_split = cast(S_split, self.dtype)
        rpms = cast(rpms, self.dtype)
        
        
        # PNG names
        make_pngname = lambda tt, rpm: self.make_pngname(x, sr, id_, *tt, rpm)
        names = list(map(make_pngname, T_split, rpms))
        
        results = {"Ss": S_split, 
                   "rpms": rpms, 
                   "names": names, 
                   "avg_rpm": avg_rpm, 
                   "overload": overload}
        
        if self._plot < 1:
            return results
        
        # Plot
        clabel = 'dB SPL' if self.SPL else 'pressure amplitude [Pa]'
        r_to_blades = {0: 'A-B-C', 1: 'B-C-A', 2: 'C-A-B'}
        
        for r, (lefts, rights) in bounds.items():
            fig, ax = plt.subplots()
            plt.suptitle(id_)
            plt.title('Spectrogram (%s)' % r_to_blades[r])
            plt.pcolormesh(T, F, S, vmin=self.vlim[0], vmax=self.vlim[1])
            b, t = plt.ylim()
            add_rec = lambda l, r: ax.add_patch(
                plt.Rectangle((T[l], b), 
                              T[r] - T[l], 
                              t-b, 
                              color='r', 
                              alpha=0.08) 
            )
            list(map(add_rec, lefts, rights))
            plt.xlim(t_start, t_stop)
            plt.suptitle(id_)
            plt.xlabel('time [sec]')
            plt.ylabel('fequency [Hz]')
            plt.colorbar(label=clabel)
            
            if self.render:
                plt.show()
            else:
                plt.close()
            if self.save:
                fpath = os.path.join(self.save_directory, 
                                     id_+'_%d.png' % next(self._idx_gen))
                save_fig(fig, fpath, newfolder=True, makedir=True)
        
        return results
    
    def make_pngname(self, x, sr, id_, t_start, t_stop, rpm=None):
        t_total = len(x) / sr + self.begin
        n_chars = len(f'{t_total:.1f}')
        t_start = ('0'*n_chars + f'{t_start:.1f}')[-n_chars:]
        t_stop = ('0'*n_chars + f'{t_stop:.1f}')[-n_chars:]
        
        if rpm is None:
            return id_ + '_(%s~%s).png' % (t_start, t_stop)
        return id_ + '_(%s~%s)_`%.2frpm`.png' % (t_start, t_stop, rpm)
    
    def spectrogram(self, x, sr, t_start) -> tuple:
        windowlength = 512    # (points)
        nfft = 512    # (points)
        noverlap = nfft // 2    # (points)
        window = signal.windows.hann(windowlength, sym=False)  # periodic
        
        S, F, T = mlab.specgram(
            x, 
            NFFT=nfft, 
            Fs=sr, 
            window=window, 
            noverlap=noverlap, 
            mode='magnitude', 
            sides='onesided'    # f >= 0
        )
        T = T - T[0] + t_start
        pps = len(T) / (T[-1] - T[0])
        
        return S, F, T, pps
    
    def detect_overload(self, S, F, T, pps, sensitivity, scale) -> dict:
        cutin_freq = 2000    # (Hz)
        threshold = 0.005    # (V)
         # When sample rate = 25600 Hz & spectrogram window size = 512 points.
        ratio = 0.03    # max acceptable overload ratio for each segment
        segment_duration = 1.5    # (sec)
        
        # Find the indices where the overloads appear
        half_seg = int(segment_duration * pps // 2)   # (points)
        wave, _ = frequency_filter(S, F=F, cutin_freq=cutin_freq)
        wave *= scale / sensitivity    # (Pa) => (V)
        wave = wave.sum(axis=0)
        
        def overratio(idx):
            segment = overloaded[idx-half_seg: idx+half_seg]
            return segment.sum() / len(segment) > ratio
        overloaded = wave < threshold
        overload = np.nonzero(overloaded)[0]
        overload = np.array(list(filter(overratio, overload)))
        
        overload = {
            "middles": overload, 
            "ratio": len(overload) / len(T)
        }
        
        self.print_fn('overload ratio: {:.4f}'.format(overload['ratio']))
        if overload['ratio'] > ratio:
            self.print_fn('**WARNING**\n'
                          'The overload ratio is higher than %s! There\'s too '
                          'much useless data in this audio file.' % ratio)
        
        return overload
    
    def estimate_rotorspeed(self, S, F, T, pps, id_, overload) -> tuple:
        cutin_freq1 = 1500    # cut-in freq of sound
        cutoff_freq1 = 2500    # cut-off freq of sound
        cutin_freq2 = 8500    # cut-in freq of sound
        cutoff_freq2 = 12500    # cut-off freq of sound
        
        min_rs, max_rs = RotorSpeeds
        
        
        # (rpm) => (rps)
        min_rs /= 60
        max_rs /= 60
        
        d0 = min_rs    # cut-in => min rotor speed (Hz)
        d1 = max_rs * 3    # cut-off => max blade speed (Hz)
        
        max_rp = 1 / min_rs    # max rotor period
        min_rp = 1 / max_rs    # min rotor period
        
        # Frequency cutting
        wave1, _ = frequency_filter(S, F, 
                                    cutin_freq=cutin_freq1, 
                                    cutoff_freq=cutoff_freq1)
        wave2, _ = frequency_filter(S, F, 
                                    cutin_freq=cutin_freq2, 
                                    cutoff_freq=cutoff_freq2)
        wave = np.concatenate([wave1, wave2], axis=0)
        
        # Mean normalize on each freq bin
        # to retain the variety of power w.r.t. time 
        max_ = wave.max(axis=1, keepdims=True)
        min_ = wave.min(axis=1, keepdims=True)
        wave -= wave.mean(axis=1, keepdims=True)
        wave /= (max_ - min_)
        
        low_freq_range = wave1.shape[0]
        wave[:low_freq_range] *= 0.5
        wave = wave.sum(axis=0)    # 2D => 1D array
        del wave1, wave2
        
        
        # Moving average
        n_points = round(pps * 1/4/d1)    # kernel size (blade period)/4
        wave_ma = moving_average(wave, n_points=n_points)
        
        if self._plot >= 3:
            # Plot the first 20 sec after moving average
            stop = np.argwhere(T >= T[0] + 20)[0][0]
            fig = plt.figure()
            plt.title('Moving average smoothing (20 sec)')
            plt.plot(T[:stop], wave[:stop])
            plt.plot(T[:stop], wave_ma[:stop], 'r', lw=1.3, 
                     label='moving average')
            plt.xlabel('time [sec]')
            plt.ylabel('magnitude')
            plt.legend()
            if self.render:
                plt.show()
            else:
                plt.close()
            if self.save:
                fpath = os.path.join(self.save_directory, 
                                     id_+'_%d.png' % next(self._idx_gen))
                save_fig(fig, fpath, newfolder=True, makedir=True)
        
        
        # Butterworth freq domain filtering
        freqs = fftfreq( len(wave_ma), d=1/pps )    # -f, +f
        LPn, HPn = 5, 3
        with np.errstate(divide='ignore'):
            LPfilter = 1 / ( 1 + (freqs/d1)**(2*LPn) )
            HPfilter = 1 / ( 1 + (d0/freqs)**(2*HPn) )
        BPfilter = LPfilter * HPfilter
        BPfilter /= BPfilter.max()  # normalize
        
        wave_bp = ifft(
            fft(wave_ma, workers=-1) * BPfilter, workers=-1
        )
        wave_bp = wave_bp.real    # remove the Imagine part
        
        
        if self._plot >= 3:
            # Plot frequency domain before & after BPF
            amp_ma = np.abs( rfft(wave_ma, workers=-1) )  # f >= 0 (symmetric)
            freqs_ma = rfftfreq( len(wave_ma), d=1/pps )
            
            amp_bp = np.abs( rfft(wave_bp, workers=-1) )  # f >= 0
            freqs_bp = rfftfreq( len(wave_bp), d=1/pps )
            
            fig, axs = plt.subplots(2, 2, sharex='col', sharey=True, 
                                    gridspec_kw={'width_ratios': [2, 1]})
            
            axs[0,0].set_title('Frequency domain (before BPF)')
            axs[0,0].plot(freqs_ma, amp_ma)
            axs[0,0].set_ylabel('magnitude')
            axs[0,0].set_xscale('log')
            
            axs[1,0].set_title('Frequency domain (after BPF)')
            axs[1,0].plot(freqs_bp, amp_bp)
            axs[1,0].set_xlabel('frequency [Hz]')
            axs[1,0].set_ylabel('magnitude')
            axs[1,0].set_xscale('log')
            
            freqs_pos = freqs[ freqs >= 0]
            filter_pos = BPfilter[ freqs >= 0]
            max_to_one = lambda x: x / amp_ma.max()    # first axis
            one_to_max = lambda y: y * amp_ma.max()    # secondary axis
            functions = (max_to_one, one_to_max)
            
            axs[0,1].plot(freqs_ma, amp_ma)
            axs[0,1].plot(freqs_pos, filter_pos * amp_ma.max(), 'r', lw=1.5)
            axs[0,1].secondary_yaxis('right', functions=functions, color='r')
            
            axs[1,1].plot(freqs_bp, amp_bp, label='magnitude of FFT')
            axs[1,1].plot(freqs_pos, 
                          filter_pos * amp_ma.max(), 
                          'r', 
                          lw=1.5, 
                          label='filter')
            axs[1,1].set_xlim(-0.1, 2)
            axs[1,1].secondary_yaxis('right', functions=functions, color='r')
            axs[1,1].set_xlabel('Frequency [Hz]')
            axs[1,1].legend()
            if self.render:
                plt.show()
            else:
                plt.close()
            if self.save:
                fpath = os.path.join(self.save_directory, 
                                     id_+'_%d.png' % next(self._idx_gen))
                save_fig(fig, fpath, newfolder=True, makedir=True)
        
        
        # Find local minima
        _minima, _ = signal.find_peaks(-wave_bp)
        diff = np.diff(_minima)
        diff = np.append(diff, diff[-1])
        
        duration = 20    # (sec)
        overlap = 5    # (sec)
        nhop = int((duration - overlap) * pps)    # (points)
        nduration = int(duration * pps)    # (points)
        n_clips = int(np.ceil( 1 + (len(wave_bp)-nduration) / (nhop) ))
        
        if self._plot >= 2:
            clabel = 'dB SPL' if self.SPL else 'Pressure Amplitude [Pa]'
            S = to_SPL(S) if self.SPL else S
        
        valleys = list()    # global indices
        prev_stop = -1
        for c in range(n_clips):
            if c == n_clips - 1:    # last clip
                stop = len(wave_bp)
                start = stop - nduration
            else:
                start = nhop * c
                stop = start + nduration
            
            wave_bp_clip = wave_bp[start:stop]
            diff_clip = diff[ (start <= _minima) & (_minima <= stop) ]
            min_distance = np.median(diff_clip) * 0.5
            valleys_clip, _ = signal.find_peaks(
                -np.insert(wave_bp_clip[[0, -1]], 1, wave_bp_clip), 
                distance=min_distance
            )    # find the local minima whose distances are larger than  
                 # `min_distance` local indices
            valleys_clip -= 1 if c > 0 else 0
            
            valleys_global = valleys_clip + start    # => global indices
            valleys += list(filter(lambda i: i >= prev_stop-1, valleys_global))
             # filter out the ones in the overlap regions
            prev_stop = stop
            
            if self._plot < 2:
                continue
            # Plot Spectrogram & overload positions
            T_clip = T[start:stop]
            S_clip = S[:, start:stop]
            wave_clip = wave[start:stop]
            wave_ma_clip = wave_ma[start:stop]
            middles_clip = list(filter(lambda x: start <= x < stop, 
                                       overload['middles']))
            
            fig, axs = plt.subplots(2, 1, sharex=True)
            plt.suptitle(id_)
            
            axs[0].set_title('Spectrogram')
            img = axs[0].pcolormesh(T_clip, F, S_clip, 
                                    vmin=self.vlim[0], vmax=self.vlim[1])
            axs[0].set_ylabel('frequency [Hz]')
            plt.colorbar(img, ax=axs[0], label=clabel)
            if middles_clip:
                ylim = axs[0].get_ylim()
                x = T[middles_clip]
                y = np.full_like(x, np.mean(ylim))
                
                axs[0].scatter(x, y, s=60, 
                               color='r', marker='x', lw=1.5, label='overload')
                axs[0].legend()
            
            # Plot rotor speed estimation
            minimum = min(*wave_ma_clip, *wave_bp_clip)
            maximum = max(*wave_ma_clip, *wave_bp_clip)
            delta = maximum - minimum
            ymin = minimum - delta * 0.1
            ymax = maximum + delta * 0.2
            
            axs[1].set_title('Rotor speed estimation')
            axs[1].plot(T_clip, wave_clip, '#00CAAA', label='power wave')
            axs[1].plot(T_clip, wave_ma_clip, 
                        '#003CCC', lw=1.2, label='moving average')
            axs[1].plot(T_clip, wave_bp_clip, 'k', lw=1.4, label='filtered')
            axs[1].vlines(T_clip[valleys_clip], ymin, ymax, 
                          'r', lw=1.5, zorder=2.1, label='local min')
            axs[1].set_xlim( round(T_clip[0]), round(T_clip[-1]) )
            axs[1].set_ylim(ymin, ymax)
            axs[1].set_xlabel('time [sec]')
            axs[1].set_ylabel('magnitude')
            axs[1].legend(ncol=4)
            if self.render:
                plt.show()
            else:
                plt.close()
            if self.save:
                fpath = os.path.join(self.save_directory, 
                                     id_+'_%d.png' % next(self._idx_gen))
                save_fig(fig, fpath, newfolder=True, makedir=True)
        
        valleys = np.unique(valleys)
        if valleys[0] == 0:
            valleys = valleys[1:]
        if valleys[-1] == len(T)-1:
            valleys = valleys[:-1]
        
        
        # 2/3 overlapping => {A-B-C, B-C-A, C-A-B}
        bounds = {0: valleys[0::3], 
                  1: valleys[1::3], 
                  2: valleys[2::3]}
        for r, rounds in bounds.items():
            lefts, rights = rounds[:-1], rounds[1:]
            bounds[r] = np.stack([lefts, rights])
        
        
        # Irrational rotor speeds
        tolerance = 0.1    # (sec)
        upper_dt = max_rp / 3 + tolerance    # (sec)
        lower_dt = min_rp / 3 - tolerance    # (sec)
        dT = np.diff( T[valleys] )    # (sec)
        irrationals = filter(
            lambda irr_dt: (irr_dt[-1] < lower_dt) or (irr_dt[-1] > upper_dt), 
            zip(valleys, dT)
        )
        irrationals = list(map(lambda irr_dt: irr_dt[0], irrationals))
        
        irrational_ratio = len(irrationals) * 2 / len(valleys)
        self.print_fn('irrational ratio: {:.4f}'.format(irrational_ratio))
        if irrational_ratio > 0.25:
            self.print_fn('**WARNING**\n'
                          'The irrational ratio is higher than 0.25! There\'s '
                          'too much noise in this audio file!')
        
        return bounds, irrationals, irrational_ratio
    
    def filter_bounds(self, bounds, overload, irrationals) -> dict:
        # 'dynamic' == 'dynamic012'
        remainders = [ int(d) for d in re.findall(r"[0-9]", self.mode) ] \
                     or [0, 1, 2]
        bounds = dict(filter(
            lambda r_bounds: r_bounds[0] in remainders, 
            bounds.items()
        ))
        
        # Remove the irrational or overloaded ones
        for r, (lefts, rights) in bounds.items():
            between = lambda middle: (lefts <= middle) & (middle <= rights)
            
            irrational = map(between, irrationals)
            irrational = np.any(list(irrational), axis=0)
            
            overloaded = map(between, overload['middles'])
            overloaded = np.any(list(overloaded), axis=0)
            
            usable = ~(irrational | overloaded)
            if usable.size > 1:
                bounds[r] = np.stack((lefts[usable], rights[usable]))
        
        return bounds
    
    def split_spectrogram(self, S, T, bounds, irrational_ratio) -> tuple:# Split the spectrogram
        S_split, T_split = list(), list()
        for lefts, rights in bounds.values():
            S_split += list(map(lambda l, r: S[:, l:r], lefts, rights))
            T_split += list(map(lambda l, r: (T[l], T[r]), lefts, rights))
        
        lefts, rights = np.asarray(T_split).T
        periods = rights - lefts
        avg_period = periods.mean()
        rpms = (60 / periods).tolist()
        avg_rpm = 60 / avg_period
        
        self.print_fn('averaged rotor period: %.2f sec/round' % avg_period)
        self.print_fn('averaged rotor speed: %.2f round/sec (%.2f rpm)' % (
            1/avg_period, avg_rpm))
        
        return S_split, T_split, rpms, avg_rpm
    
    @property
    def _plot(self):
        return self.plot if self.save or self.render else 0



class Generator:
    def __init__(self, start=0, stop=float('+inf'), step=1):
        assert start is not None, start
        self._generator = self.build(start, stop, step)
        self.start = start
        self.stop = stop
        self.step = step
    
    def build(self, index, stop, step):
        if step > 0:
            stop_criterion_not_met = lambda index: index < stop
        else:
            stop_criterion_not_met = lambda index: index > stop
        
        while stop_criterion_not_met(index):
            yield index
            index += step
    
    def __next__(self):
        return next(self._generator)
    
    def __iter__(self):
        return self._generator


