# George Miloshevich 2021
# Importation des librairies
from pathlib import Path
from netCDF4 import Dataset
import xarray as xr
import numpy as np
import warnings

import matplotlib.pyplot as plt
import pylab as p
import sys
import os
import logging

import matplotlib.patheffects as PathEffects
from matplotlib.transforms import Bbox

from itertools import chain
import collections
from random import randrange
import pandas as pd

from scipy.signal import argrelextrema
from scipy.stats import skew, kurtosis
from scipy import integrate
from scipy.optimize import curve_fit

from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from skimage.transform import resize

path_to_parent = str(Path(__file__).resolve().parent.parent)
if not path_to_parent in sys.path:
    sys.path.insert(1, path_to_parent)

try:
    import general_purpose.utilities as ut
except ImportError:
    import ERA.utilities as ut

logger = logging.getLogger(__name__)
logger.level = logging.INFO


global plotter
plotter = None


def _sliding_window_view_dispatcher(x, window_shape, axis=None, *,
                                    subok=None, writeable=None):
    return (x,)

from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.stride_tricks import array_function_dispatch,_maybe_view_as_subclass,as_strided
# The functions below are not necessary if you are using numpy>=1.20
# To be able to use sliding_window_view, you can load from
# ef.sliding_window_view
@array_function_dispatch(_sliding_window_view_dispatcher)
def sliding_window_view(x, window_shape, axis=None, *,
                        subok=False, writeable=False):
    """
    Create a sliding window view into the array with the given window shape.
    Also known as rolling or moving window, the window slides across all
    dimensions of the array and extracts subsets of the array at all window
    positions.
    
    .. versionadded:: 1.20.0
    Parameters
    ----------
    x : array_like
        Array to create the sliding window view from.
    window_shape : int or tuple of int
        Size of window over each axis that takes part in the sliding window.
        If `axis` is not present, must have same length as the number of input
        array dimensions. Single integers `i` are treated as if they were the
        tuple `(i,)`.
    axis : int or tuple of int, optional
        Axis or axes along which the sliding window is applied.
        By default, the sliding window is applied to all axes and
        `window_shape[i]` will refer to axis `i` of `x`.
        If `axis` is given as a `tuple of int`, `window_shape[i]` will refer to
        the axis `axis[i]` of `x`.
        Single integers `i` are treated as if they were the tuple `(i,)`.
    subok : bool, optional
        If True, sub-classes will be passed-through, otherwise the returned
        array will be forced to be a base-class array (default).
    writeable : bool, optional
        When true, allow writing to the returned view. The default is false,
        as this should be used with caution: the returned view contains the
        same memory location multiple times, so writing to one location will
        cause others to change.
    Returns
    -------
    view : ndarray
        Sliding window view of the array. The sliding window dimensions are
        inserted at the end, and the original dimensions are trimmed as
        required by the size of the sliding window.
        That is, ``view.shape = x_shape_trimmed + window_shape``, where
        ``x_shape_trimmed`` is ``x.shape`` with every entry reduced by one less
        than the corresponding window size.
    See Also
    --------
    lib.stride_tricks.as_strided: A lower-level and less safe routine for
        creating arbitrary views from custom shape and strides.
    broadcast_to: broadcast an array to a given shape.
    Notes
    -----
    For many applications using a sliding window view can be convenient, but
    potentially very slow. Often specialized solutions exist, for example:
    - `scipy.signal.fftconvolve`
    - filtering functions in `scipy.ndimage`
    - moving window functions provided by
      `bottleneck <https://github.com/pydata/bottleneck>`_.
    As a rough estimate, a sliding window approach with an input size of `N`
    and a window size of `W` will scale as `O(N*W)` where frequently a special
    algorithm can achieve `O(N)`. That means that the sliding window variant
    for a window size of 100 can be a 100 times slower than a more specialized
    version.
    Nevertheless, for small window sizes, when no custom algorithm exists, or
    as a prototyping and developing tool, this function can be a good solution.
    Examples
    --------
    >>> x = np.arange(6)
    >>> x.shape
    (6,)
    >>> v = sliding_window_view(x, 3)
    >>> v.shape
    (4, 3)
    >>> v
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    This also works in more dimensions, e.g.
    >>> i, j = np.ogrid[:3, :4]
    >>> x = 10*i + j
    >>> x.shape
    (3, 4)
    >>> x
    array([[ 0,  1,  2,  3],
           [10, 11, 12, 13],
           [20, 21, 22, 23]])
    >>> shape = (2,2)
    >>> v = sliding_window_view(x, shape)
    >>> v.shape
    (2, 3, 2, 2)
    >>> v
    array([[[[ 0,  1],
             [10, 11]],
            [[ 1,  2],
             [11, 12]],
            [[ 2,  3],
             [12, 13]]],
           [[[10, 11],
             [20, 21]],
            [[11, 12],
             [21, 22]],
            [[12, 13],
             [22, 23]]]])
    The axis can be specified explicitly:
    >>> v = sliding_window_view(x, 3, 0)
    >>> v.shape
    (1, 4, 3)
    >>> v
    array([[[ 0, 10, 20],
            [ 1, 11, 21],
            [ 2, 12, 22],
            [ 3, 13, 23]]])
    The same axis can be used several times. In that case, every use reduces
    the corresponding original dimension:
    >>> v = sliding_window_view(x, (2, 3), (1, 1))
    >>> v.shape
    (3, 1, 2, 3)
    >>> v
    array([[[[ 0,  1,  2],
             [ 1,  2,  3]]],
           [[[10, 11, 12],
             [11, 12, 13]]],
           [[[20, 21, 22],
             [21, 22, 23]]]])
    Combining with stepped slicing (`::step`), this can be used to take sliding
    views which skip elements:
    >>> x = np.arange(7)
    >>> sliding_window_view(x, 5)[:, ::2]
    array([[0, 2, 4],
           [1, 3, 5],
           [2, 4, 6]])
    or views which move by multiple elements
    >>> x = np.arange(7)
    >>> sliding_window_view(x, 3)[::2, :]
    array([[0, 1, 2],
           [2, 3, 4],
           [4, 5, 6]])
    A common application of `sliding_window_view` is the calculation of running
    statistics. The simplest example is the
    `moving average <https://en.wikipedia.org/wiki/Moving_average>`_:
    >>> x = np.arange(6)
    >>> x.shape
    (6,)
    >>> v = sliding_window_view(x, 3)
    >>> v.shape
    (4, 3)
    >>> v
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    >>> moving_average = v.mean(axis=-1)
    >>> moving_average
    array([1., 2., 3., 4.])
    Note that a sliding window approach is often **not** optimal (see Notes).
    """
    window_shape = (tuple(window_shape)
                    if np.iterable(window_shape)
                    else (window_shape,))
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array < 0):
        raise ValueError('`window_shape` cannot contain negative values')

    if axis is None:
        axis = tuple(range(x.ndim))
        if len(window_shape) != len(axis):
            raise ValueError(f'Since axis is `None`, must provide '
                             f'window_shape for all dimensions of `x`; '
                             f'got {len(window_shape)} window_shape elements '
                             f'and `x.ndim` is {x.ndim}.')
    else:
        axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
        if len(window_shape) != len(axis):
            raise ValueError(f'Must provide matching length window_shape and '
                             f'axis; got {len(window_shape)} window_shape '
                             f'elements and {len(axis)} axes elements.')

    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

    # note: same axis can be windowed repeatedly
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError(
                'window shape cannot be larger than input array shape')
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape
    return as_strided(x, strides=out_strides, shape=out_shape,
                      subok=subok, writeable=writeable)

def _broadcast_to(array, shape, subok, readonly):
    shape = tuple(shape) if np.iterable(shape) else (shape,)
    array = np.array(array, copy=False, subok=subok)
    if not shape and array.shape:
        raise ValueError('cannot broadcast a non-scalar to a scalar array')
    if any(size < 0 for size in shape):
        raise ValueError('all elements of broadcast shape must be non-'
                         'negative')
    extras = []
    it = np.nditer(
        (array,), flags=['multi_index', 'refs_ok', 'zerosize_ok'] + extras,
        op_flags=['readonly'], itershape=shape, order='C')
    with it:
        # never really has writebackifcopy semantics
        broadcast = it.itviews[0]
    result = _maybe_view_as_subclass(array, broadcast)
    # In a future version this will go away
    if not readonly and array.flags._writeable_no_warn:
        result.flags.writeable = True
        result.flags._warn_on_write = True
    return result


def import_basemap():
    old_proj_lib = os.environ['PROJ_LIB'] if 'PROJ_LIB' in os.environ else None
    try:
        os.environ['PROJ_LIB'] = '../usr/share/proj' # This one we need to import Basemap 
        global Basemap
        from mpl_toolkits.basemap import Basemap
        logger.info('Successfully imported basemap')
        return True
    except (ImportError, FileNotFoundError):
        # revert to old proj_lib
        if old_proj_lib is not None:
            os.environ['PROJ_LIB'] = old_proj_lib
        logger.warning('In this environment you cannot import Basemap')
        return False
    
def import_cartopy():
    try:
        global cplt
        try:
            import general_purpose.cartopy_plots as cplt
        except ImportError:
            import cartopy_plots as cplt
        logger.info('Successfully imported cartopy')
        return True
    except (ImportError, FileNotFoundError):
        logger.warning('In this environment you cannot import cartopy')
        return False

# set up the plotter:
def setup_plotter():
    global plotter
    if plotter is not None:
        logger.info(f'Plotter already set to {plotter}')
        return True
    logger.info('Trying to import basemap')
    if import_basemap():
        plotter = 'basemap'
        return True
    logger.info('Trying to import cartopy')
    if import_cartopy():
        plotter = 'cartopy'
        return True
    logger.error('No valid plotter found')
    return False
            
setup_plotter()


# Definition des fonctions
def significative_data(Data, Data_t_value=None, T_value=None, both=False, default_value=0): # CHANGE THIS FOR TEMPERATURE SO THAT THE OLD ROUTINE IS USED
    '''
    Filters `Data` depending whether `Data_t_value` exceeds a threshold `T_value`
    
    Data that fail the filter conditions are set to `default_value`.
    If `Data_t_value` or `T_value` are None, all of `Data` is considered significant
    '''
    return ut.significative_data(Data, Data_t_value, T_value, both=both, default_value=default_value)
    
def significative_data2(Data, Data_t_value, T_value, both): # CHANGE THIS FOR TEMPERATURE SO THAT THE OLD ROUTINE IS USED
    '''
    Does the same of significative_data, but with `default_value` to np.NaN
    '''
    return ut.significative_data(Data, Data_t_value, T_value, both=both, default_value=np.NaN)
    # OLD VERSION
    
    # Out_taken = np.empty((np.shape(Data)))
    # Out_taken[:] = np.NaN
    # Out_not_taken = np.empty((np.shape(Data)))
    # Out_not_taken[:] = np.NaN
    # N_points_taken = 0
    # for la in range(len(Data)):
    #     for lo in range(len(Data[la])):
    #         if abs(Data_t_value[la, lo]) >= T_value:
    #             Out_taken[la, lo] = Data[la, lo]
    #             N_points_taken += 1
    #         else:
    #             Out_not_taken[la, lo] = Data[la, lo]
    # if both == True:
    #     return Out_taken, Out_not_taken, N_points_taken
    # elif both == False:
    #     return Out_taken, N_points_taken
    
def animate(i, m, Center_map, Nb_frame, Lon, Lat, T_value, data_colorbar_value, data_colorbar_t, data_colorbar_level,
            data_contour_value, data_contour_t, data_contour_level, title_frame, rtime):
    '''
    Center_map not used
    
    Plots at day `i` the contourf of the sigificant temperature and contour of the geopotential separating significant and non significant anomalies
    '''
    if plotter == 'cartopy':
        raise NotImplementedError("Use cartopy_plots.animate")
    fmt = '%1.0f'
    temp_sign, ts_taken = significative_data(data_colorbar_value[i], data_colorbar_t[i], T_value, False)
    zg_sign, zg_not, zg_taken = significative_data2(data_contour_value[i], data_contour_t[i], T_value, True)
    if ts_taken != 8192 and zg_taken != 8192: #AL what is this???
        print('i:', i, 'ts_taken:', ts_taken, 'zg_taken:', zg_taken)
    plt.cla()
    m.contourf(Lon, Lat, temp_sign, levels=data_colorbar_level, cmap=plt.cm.seismic, extend='both', latlon=True)
    m.colorbar()
    c_sign = m.contour(Lon, Lat, temp_sign, levels=data_colorbar_level, colors="black",linewidths=1, latlon=True, linestyles = "dotted")
    m.drawcoastlines(color='black',linewidth=1)
    m.drawparallels(np.arange(-80.,81.,20.),linewidth=0.5,labels=[True,False,False,False], color = "green")
    m.drawmeridians(np.arange(-180.,181.,20.),linewidth=0.5,labels=[False,False,False,True],color = "green")
    
    
    c_nots = m.contour(Lon, Lat, data_contour_value[i], levels=data_contour_level[:data_contour_level.shape[0]//2], colors="chocolate", linestyles = "dashed", linewidths=1, latlon=True) #negative insignificant anomalies of geopotential
    v_sign = data_contour_level[int(len(data_contour_level) / 2)-1], # data_contour_level[int(len(data_contour_level) / 2)]
    if len(c_nots.levels) > len(v_sign):
        p.clabel(c_nots, v_sign, inline=True,fmt = fmt,fontsize=12)
    c_nots = m.contour(Lon, Lat, data_contour_value[i], levels=data_contour_level[data_contour_level.shape[0]//2:], colors="tab:blue", linestyles = "dashed",linewidths=1, latlon=True)  #positive insignificant anomalies of geopotential
    v_sign = data_contour_level[int(len(data_contour_level) / 2)],
    if len(c_nots.levels) > len(v_sign):
        p.clabel(c_nots, v_sign, inline=True,fmt = fmt,fontsize=12)
    
    c_sign = m.contour(Lon, Lat, zg_sign, levels=data_contour_level[:data_contour_level.shape[0]//2], colors="red", linestyles = "solid",linewidths=1, latlon=True)  #negative significant anomalies of geopotential
    c_sign = m.contour(Lon, Lat, zg_sign, levels=data_contour_level[data_contour_level.shape[0]//2:], colors="blue",linewidths=1, latlon=True)   #positive significant anomalies of geopotential
    
    plt.title(f'{title_frame}, r = {rtime}, day: {(i - Nb_frame//2)}', fontsize=20)

    
def PltAnomalyHist(distrib, numlevels, mycolor, myhatch, mymonths, mylinewidths, myfieldlabel, myobjct): # Plot histogram of an anomaly based on a data series
    # histogram
    n, bins, patches = plt.hist(distrib, 50, density=True, histtype='step', color=mycolor, hatch=myhatch,
                                alpha=1, label=r"{:s}:  std = {:1.3f}, skew = {:1.3f}, kurt = {:1.3f}".format(
                                    mymonths,np.std(distrib),skew(distrib),kurtosis(distrib, fisher=True)),
                                linewidth=mylinewidths)
    # plot gaussian approximation of the anomaly
    plt.plot(bins,np.exp(-bins**2/(2*np.std(distrib)**2))/(np.std(distrib)*np.sqrt(2*np.pi)),
             color = mycolor,linestyle='dashed')
    plt.yscale("log")
    plt.xlabel(myfieldlabel + ' detrended anomaly')
    plt.ylabel('Daily Probability')
    plt.title('MMJAS (running mean) ' + myobjct)
    
def PltDistHist(myfield, convseq, mycolors, monthhatches, mymonth, mylinewidth, y, Tot_Mon1,area, start_month=5):
    # Plot Distribution for each year and histograms for the anomalies
    field_extract = myfield.abs_area_int[:,:]
    objct = "over "+area
    plt.figure(figsize=(30,5))
    plt.subplot(141)
    plt.plot(field_extract[y],label = 'daily')
    plt.plot(np.convolve(field_extract[y], convseq, mode='valid'),label = f'{len(convseq)} d mean')
    plt.title(f'Time evolution {objct} in {y = }')
    plt.xlabel('Days from May 0')
    plt.ylabel(myfield.label)
    plt.legend(loc='best')

    color_idx = np.linspace(0, 1, myfield.var.shape[0])
    plt.subplot(142)
    field_conv = []
    for year in range(myfield.var.shape[0]):
        field_conv.append(np.convolve(field_extract[year], convseq, mode='valid'))
        plt.plot(field_conv[year],color=plt.cm.rainbow(color_idx[year]))
    field_conv = np.array(field_conv)
    plt.plot(np.convolve(field_extract[y], convseq, mode='valid'),'k-.')
    plt.title(f'{len(convseq)} day running mean {objct}')
    plt.xlabel('Days from May 0')
    plt.ylabel(myfield.label)

    plt.subplot(143)
    for i in range(len(mymonth)):
        temp = field_extract[:,Tot_Mon1[start_month+i]-Tot_Mon1[start_month]:Tot_Mon1[start_month+1+i]-Tot_Mon1[start_month]].reshape(myfield.var.shape[0]*(Tot_Mon1[start_month+1+i]-Tot_Mon1[start_month+i]))
        PltAnomalyHist(temp, 50, mycolors[i], monthhatches[i], mymonth[i], mylinewidth[i], myfield.label, objct)
    plt.legend(loc = 'best')

    plt.subplot(144)
    for i in range(len(mymonth)-1): # Here you have to be careful because A(t) is not defined for the september
        temp = field_conv[:,Tot_Mon1[start_month+i]-Tot_Mon1[start_month]:Tot_Mon1[start_month+1+i]-Tot_Mon1[start_month]].reshape(myfield.var.shape[0]*(Tot_Mon1[start_month+1+i]-Tot_Mon1[start_month+i])) # 30 DAYS MAY NOT WORK FOR CESM
        PltAnomalyHist(temp, 50, mycolors[i], monthhatches[i], mymonth[i], mylinewidth[i], myfield.label, objct)
    plt.legend(loc = 'best')
    
def PltReturnsHist(XX_rt, YY_rt, xx_rt, yy_rt, A_max_sorted, Tot_Mon1, area, ax1, Ax, start_month=5, end_month=8):
    # Plot Return times plus the histogram during the 
    ax1.scatter(XX_rt, YY_rt, s=4, color='royalblue', marker='x')
    for i in range(len(xx_rt)):
        ax1.text(xx_rt[i] + 0.02, -4.2, 'a{}={:.2f}'.format(int(xx_rt[i]), yy_rt[i]))
        ax1.plot([xx_rt[i], xx_rt[i]], [-4.5, yy_rt[i]], linestyle='--', color='black', linewidth=0.5)
    ax1.set_xscale('log')
    ax1.set_xlabel('return time $\hat{r}$ (year)')
    ax1.set_ylabel('temperature anomaly threshold $a_r$ (K)')
    ax1.set_title('Temperature anomalies over '+ area, loc='left')
    Days = []
    years = []
    for i in range(len(A_max_sorted)):
        day, year = A_max_sorted[i][1]   # heatwaves are already ranked by Phlippine based on 14 day temperature anomalies (Notice that she counts from June 1!)
        Days.append(day+Tot_Mon1[start_month])
        years.append(year)
    
    # top 1/10 extreme events
    n, bins, patches = Ax.hist(Days[:len(A_max_sorted)//10], bins = np.arange(Tot_Mon1[start_month],Tot_Mon1[end_month]-14),
                               density = True, facecolor='tab:brown', alpha=1, label = 'extreme $r=10$')
    # top 1/4 extreme events
    Ax.hist(Days[:len(A_max_sorted)//4], bins = np.arange(Tot_Mon1[start_month],Tot_Mon1[end_month]-14),
            density = True, facecolor='tab:orange', alpha=0.7, label = 'extreme $r=4$')
    # all extreme events
    Ax.hist(Days[:len(A_max_sorted)], bins = np.arange(Tot_Mon1[start_month],Tot_Mon1[end_month]-14),
            density = True, facecolor='tab:cyan', alpha=0.3, label = 'extreme $r=1$')
    Ax.set_xlabel('Time, binned by {:1.4f}'.format(bins[1]-bins[0]), fontsize=13)
    Ax.set_ylabel('Probability', fontsize=14)
    Ax.set_ylim([0, 0.75*np.max(n)])
    Ax.set_title("Events conditioned", fontsize=13)
    Ax.legend(loc = 'best', fontsize=12)
    
def BootstrapReturnsOnly(myseries, TO, Tot_Mon1, area, ax, Ts, modified='no', write_path='./', start_month=6, end_month=9):
    write_path = write_path.rstrip('/')
    
    for T in Ts:
        convseq = np.ones(T)/T
        XX = []
        YY = []
        for j in range(10):
            A = np.zeros((myseries.shape[0]//10, Tot_Mon1[end_month] - Tot_Mon1[start_month] - T+1))   # When we use convolve (running mean) there is an extra point that we can generate by displacing the window hence T-1 instead of T
            for y in range(myseries.shape[0]//10):
                A[y,:] = np.convolve(myseries[100*j+y,Tot_Mon1[start_month]:(Tot_Mon1[end_month])],  convseq, mode='valid')
            print(f"{A.shape = }")
            if A.shape[1] > 30: 
                A_max, Ti, year_a = a_max_and_ti_postproc(A, A.shape[1])
            else: # The season length is too short and we need to just take maximum
                A_max = list(np.max(A,1))
                Ti = list(np.argmax(A,1))
                year_a = list(np.arange(myseries.shape[0]//10))
            #print(Ti)
            A_max_sorted = a_decrese(A_max, Ti, year_a)
            #print(A_max_sorted)
            #print(A_max_sorted)
            XX_rt, YY_rt, xx_rt, yy_rt = return_time_fix(A_max_sorted, modified)
            YY.append(np.array(YY_rt))
        YY = np.array(YY)
        
        # save results
        np.save(f'{write_path}/Postproc/{TO}_{area}_XX_rt_{T}',XX_rt)
        np.save(f'{write_path}/Postproc/{TO}_{area}_YY_mean_{T}',np.mean(YY,0))
        np.save(f'{write_path}/Postproc/{TO}_{area}_YY_std_{T}',np.std(YY,0))
        
        # plot
        plt.fill_between(XX_rt, np.mean(YY,0)-np.std(YY,0), np.mean(YY,0)+np.std(YY,0),label=f'{T} days ({TO})')
    ax.set_xscale('log')
    ax.set_xlabel('return time $\hat{r}$ (year)')
    ax.set_ylabel('temperature anomaly threshold $a_r$ (K)')
    ax.set_title(f'Temperature anomalies over {area}', loc='left')
    
def BootstrapReturns(myseries,FROM, TO, Tot_Mon1,area, ax, Ts, modified='no', read_path='./', write_path='./'):
    # remove extra / from paths
    read_path = read_path.rstrip('/')
    write_path = write_path.rstrip('/')
    
    # Compare bootstrapped method (to be saved in "TO") to the points in "FROM"
    BootstrapReturnsOnly(myseries, TO, Tot_Mon1, area, ax, Ts, modified, write_path=write_path)
    for T in Ts:
        ERA_XX_rt = np.load(f'{read_path}/../ERA/Postproc/{FROM}_{area}_XX_rt_{T}.npy')
        ERA_YY_rt = np.load(f'{read_path}/../ERA/Postproc/{FROM}_{area}_YY_rt_{T}.npy')
        plt.scatter(ERA_XX_rt, ERA_YY_rt, s=10, marker='x',label=str(T)+' days ('+FROM+')')
    ax.legend(loc='best')

def func1(x, a, b, c, d):
     return a * np.exp(b * x) + c * np.exp(d * x)
def func2(x, a, b, c, d):
     return a * np.exp(b * x)/ b + c * np.exp(d * x)/ d
    
def autocorrelation(myseries, maxlag=100):
    # this pads each year with padsize sample time of 0s so that when the array is permuted to be multiplied by itself we don't end up using the previous part of the year
    series_pad = np.pad(myseries,((0, 0), (0, maxlag)), 'constant')  
    autocorr = []
    for k in range(maxlag):
        autocorr.append(np.sum(series_pad*np.roll(series_pad, -k))/(series_pad.shape[0]*(series_pad.shape[1]-k-maxlag)))
    return autocorr

def PltAutocorrelationFit(autocorr_mean,autocorr_std,x1,x2, ax,period_label):
    '''
    ax is not used
    '''
    # Plot normalized Autocorrelation function
    plt.plot(np.arange(0, len(autocorr_mean)), autocorr_mean,'b:', label=r'$\int C(t) dt$ = %5.1f d'%(integrate.cumtrapz(np.array(autocorr_mean), range(len(autocorr_mean)), initial=0)[-1]))
    plt.plot(np.arange(0, len(autocorr_mean)), -autocorr_mean,'k:', label='- autocorrelation')
    
    popt1, pcov1 = curve_fit(func1, np.array(range(x1,x2)), autocorr_mean[x1:x2], p0=(1, 1e-6, 1, 1e-6))
    xaxis1 = np.arange(x1,x2)
    plt.plot(xaxis1, func1(xaxis1, *popt1), 'r--', label=r'-slope$^{-1}_1$ = %5.1f ,-slope$^{-2}_2$ = %5.1f ,$\quad \tau$ = %5.1f ' %(-popt1[1]**(-1),-popt1[3]**(-1),- func2(0, *popt1)))

    plt.title(period_label+r": $\Sigma_y \Sigma_t  \alpha(t) \alpha(t+\tau)/N(\tau)$")
    plt.yscale("log")
    plt.xlabel(r"Lag $\tau$")
    
def PltAutocorrelationFit2(autocorr_mean,colors, linewidths, x1,x2, ax, Model):
    # Plot normalized Autocorrelation function
    plt.plot(np.arange(0, len(autocorr_mean)), autocorr_mean,colors[0], linewidth = linewidths[0], label=Model)
    #r'$\int C(t) dt$ = %5.1f d'%(integrate.cumtrapz(np.array(autocorr_mean), range(len(autocorr_mean)), initial=0)[-1]))
    #plt.plot(np.arange(0, len(autocorr_mean)), -autocorr_mean,'k:', label='- autocorrelation')
    
    popt1, pcov1 = curve_fit(func1, np.array(range(x1,x2)), autocorr_mean[x1:x2], p0=(1, 1e-6, 1, 1e-6))
    xaxis1 = np.arange(x1,x2)
    plt.plot(xaxis1, func1(xaxis1, *popt1), colors[1], linewidth=linewidths[1],
             label=r'A= %5.3f,$\quad$ B= %5.3f,$\quad$-slope$^{-1}_1$ = %5.1f ,$\quad$-slope$^{-2}_2$ = %5.1f  ' %(popt1[0],popt1[2],-popt1[1]**(-1),-popt1[3]**(-1)))
    plt.yscale("log")
    plt.xlabel(r"Lag $\tau$")

    
def CompCompositesERA(series, myfield, T, Tot_Mon1, return_index, myfieldmean, modified='no', start_month=6, end_month=9):
    # Computes composites conditioned to extremes of field of duration T based on months provided in Tot_Mon1, the return_index is the index of the return times
    convseq = np.ones(T)/T
    A = np.zeros((series.shape[0], Tot_Mon1[end_month] - Tot_Mon1[start_month] - T+1))   # When we use convolve (running mean) there is an extra point that we can generate by displacing the window hence 13 instead of 14
    for y in range(series.shape[0]):
        A[y,:]=np.convolve(series[y,Tot_Mon1[start_month]:(Tot_Mon1[end_month])],  convseq, mode='valid')
    print("A.shape = ",A.shape)
    A_max, Ti, year_a = a_max_and_ti_postproc(A, A.shape[1])
    year_a = range(series.shape[0])
    A_max_sorted = a_decrese(A_max, Ti, year_a)
    XX_rt, YY_rt, xx_rt, yy_rt = return_time_fix(A_max_sorted, modified)
    print(xx_rt,yy_rt)

    tau = np.arange(-30,30,1)
    nb_events = 0
    myfield.composite_mean = np.zeros((len(tau),myfield.var.shape[2],myfield.var.shape[3]))
    myfield.composite_std = np.zeros((len(tau),myfield.var.shape[2],myfield.var.shape[3]))
    for y in range(series.shape[0]):
        if A_max[y] >= yy_rt[return_index]:
            print("A_max["+str(y)+"] = ",A_max[y], "Ti["+str(y)+"] = ", Ti[y])
            nb_events += 1
            value = (myfield.detrended[y] - myfieldmean)[tau + (Tot_Mon1[start_month] + Ti[y]) ]
            myfield.composite_mean += value     # This is the raw sum
            myfield.composite_std += value**2   # This is the raw square sum
    print("number of events: ",nb_events)
    # std and mean are computed below
    myfield.composite_std = np.sqrt((myfield.composite_std - (myfield.composite_mean * myfield.composite_mean / nb_events)) / (nb_events - 1))
    myfield.composite_mean /= nb_events
    myfield.composite_t = (lambda a, b: np.divide(a, b, out=np.zeros(a.shape), where=b != 0))(np.sqrt(nb_events) * myfield.composite_mean, myfield.composite_std)
    
def CompCompositesERAThreshold(series, myfield, T, Tot_Mon1, threshold, myfieldmean, start_month=6, end_month=9):
    A_max, Ti, year_a = CompExtremes(series, myfield, T, Tot_Mon1, threshold, start_month=start_month, end_month=end_month)

    tau = np.arange(-30,30,1)
    nb_events = 0
    myfield.composite_mean = np.zeros((len(tau),myfield.var.shape[2],myfield.var.shape[3]))
    myfield.composite_std = np.zeros((len(tau),myfield.var.shape[2],myfield.var.shape[3]))
    for y in range(series.shape[0]):
        if A_max[y] >= threshold:
            print("A_max["+str(y)+"] = ",A_max[y], "Ti["+str(y)+"] = ", Ti[y])
            nb_events += 1
            value = (myfield.detrended[y] - myfieldmean)[tau + (Tot_Mon1[start_month] + Ti[y]) ]
            myfield.composite_mean += value     # This is the raw sum
            myfield.composite_std += value**2   # This is the raw square sum
    print("number of events: ",nb_events)
    # std and mean are computed below
    myfield.composite_std = np.sqrt((myfield.composite_std - (myfield.composite_mean * myfield.composite_mean / nb_events)) / (nb_events - 1))
    myfield.composite_mean /= nb_events
    myfield.composite_t = (lambda a, b: np.divide(a, b, out=np.zeros(a.shape), where=b != 0))(np.sqrt(nb_events) * myfield.composite_mean, myfield.composite_std)
    
    
def CompExtremes(series, myfield, T, Tot_Mon1, threshold, start_month=6, end_month=9):
    '''
    !!!!
    myfield, threshold are not used
    !!!!
    
    The computations are performed between `start_month` (included) and `end_month` (excluded).
    Month numeration is the standard one, i.e. 1 = January, 6 = June, 12 = December
    '''
    # Computes composites conditioned to extremes of field of duration T based on months provided in Tot_Mon1, the return_index is the index of the return times
    convseq = np.ones(T)/T
    A = np.zeros((series.shape[0], Tot_Mon1[end_month] - Tot_Mon1[start_month] - T+1)) # When we use convolve (running mean) there is an extra point that we can generate by displacing the window hence T - 1 instead of T
    for y in range(series.shape[0]):
        A[y,:]=np.convolve(series[y,Tot_Mon1[start_month]:(Tot_Mon1[end_month])],  convseq, mode='valid')
    print("A.shape = ",A.shape)
    return a_max_and_ti_postproc(A, A.shape[1])

def CompCompositesThreshold(series, myfield, T, Tot_Mon1, threshold, start_month=6, end_month=9, observation_time=30, return_time_series=False):
    '''
    If `return_time_series` is true, then the time series are returned. All other computations are carried out anyways.
    `time_series` is a dictionary of the time series of `myfield` aroud the heatwaves keyed with the year number
    
    The computations are performed between `start_month` (included) and `end_month` (excluded).
    Month numeration is the standard one, i.e. 1 = January, 6 = June, 12 = December
    '''
    A_max, Ti, year_a = CompExtremes(series, myfield, T, Tot_Mon1, threshold, start_month=start_month, end_month=end_month)
    
    tau = np.arange(-observation_time,observation_time,1) # from observation_time days before to observation_time - 1  days after the heatwave
    if return_time_series:
        time_series = {}
    
    nb_events = 0
    myfield.composite_mean = np.zeros((len(tau),myfield.var.shape[2],myfield.var.shape[3])) # shape: (days, lat, lon)
    myfield.composite_std = np.zeros((len(tau),myfield.var.shape[2],myfield.var.shape[3]))
    
    # do statistics over the years
    for y in range(series.shape[0]):
        if A_max[y] >= threshold:
            print(f'A_max[{y}] = {A_max[y]}, Ti[{y}] = {Ti[y]}')
            nb_events += 1
            value = (myfield.var[y])[tau + (Tot_Mon1[start_month] + Ti[y]) ] # value of the field (over the Earth) around the days when the heatwave is at its maximum
            if return_time_series:
                time_series[y] = value
            myfield.composite_mean += value     # This is the raw sum
            myfield.composite_std += value**2   # This is the raw square sum
    print(f'number of events: {nb_events}')
    # std and mean are computed below
    myfield.composite_std = np.sqrt((myfield.composite_std - (myfield.composite_mean**2 / nb_events)) / (nb_events - 1))
    myfield.composite_mean /= nb_events
    # t = sqrt(nb_events)*composite_mean/composite_std
    myfield.composite_t = (lambda a, b: np.divide(a, b, out=np.zeros(a.shape), where=b != 0))(np.sqrt(nb_events) * myfield.composite_mean, myfield.composite_std)
    if return_time_series:
        return time_series
    else:
        return None
    
def CompComposites(series, myfield, T, Tot_Mon1, return_index, modified, start_month=6, end_month=9):
    '''
    Computes composites conditioned to extremes of field of duration T based on months provided in Tot_Mon1, the return_index is the index of the return times
    
    The computations are performed between `start_month` (included) and `end_month` (excluded).
    Month numeration is the standard one, i.e. 1 = January, 6 = June, 12 = December
    '''
    convseq = np.ones(T)/T
    A = np.zeros((series.shape[0], Tot_Mon1[end_month] - Tot_Mon1[start_month] - T+1))   # When we use convolve (running mean) there is an extra point that we can generate by displacing the window hence 13 instead of 14
    for y in range(series.shape[0]):
        A[y,:]=np.convolve(series[y,Tot_Mon1[start_month]:(Tot_Mon1[end_month])],  convseq, mode='valid')
    print("A.shape = ",A.shape)
    A_max, Ti, year_a = a_max_and_ti_postproc(A, A.shape[1])
    year_a = range(series.shape[0])
    A_max_sorted = a_decrese(A_max, Ti, year_a)
    XX_rt, YY_rt, xx_rt, yy_rt = return_time_fix(A_max_sorted, modified)
    print(xx_rt,yy_rt)

    tau = np.arange(-30,30,1)
    nb_events = 0
    myfield.composite_mean = np.zeros((len(tau),myfield.var.shape[2],myfield.var.shape[3]))
    myfield.composite_std = np.zeros((len(tau),myfield.var.shape[2],myfield.var.shape[3]))
    for y in range(series.shape[0]):
        if A_max[y] >= yy_rt[return_index]:
            print("A_max["+str(y)+"] = ",A_max[y], "Ti["+str(y)+"] = ", Ti[y])
            nb_events += 1
            value = (myfield.var[y])[tau + (Tot_Mon1[start_month] + Ti[y]) ]
            myfield.composite_mean += value     # This is the raw sum
            myfield.composite_std += value**2   # This is the raw square sum
    print("number of events: ",nb_events)
    # std and mean are computed below
    myfield.composite_std = np.sqrt((myfield.composite_std - (myfield.composite_mean * myfield.composite_mean / nb_events)) / (nb_events - 1))
    myfield.composite_mean /= nb_events
    myfield.composite_t = (lambda a, b: np.divide(a, b, out=np.zeros(a.shape), where=b != 0))(np.sqrt(nb_events) * myfield.composite_mean, myfield.composite_std)
    
def CompCompositesBetween(series, myfield, T, Tot_Mon1, return_index, start_month=6, end_month=9):
    '''
    Computes composites conditioned to extremes of field of duration T based on months provided in Tot_Mon1, the return_index is the index of the return times
    
    The computations are performed between `start_month` (included) and `end_month` (excluded).
    Month numeration is the standard one, i.e. 1 = January, 6 = June, 12 = December
    '''
    convseq = np.ones(T)/T
    A = np.zeros((series.shape[0], Tot_Mon1[end_month] - Tot_Mon1[start_month] - T+1))   # When we use convolve (running mean) there is an extra point that we can generate by displacing the window hence 13 instead of 14
    for y in range(series.shape[0]):
        A[y,:]=np.convolve(series[y,Tot_Mon1[start_month]:(Tot_Mon1[end_month])],  convseq, mode='valid')
    print("A.shape = ",A.shape)
    A_max, Ti, year_a = a_max_and_ti_postproc(A, A.shape[1])
    year_a = range(series.shape[0])
    A_max_sorted = a_decrese(A_max, Ti, year_a)
    XX_rt, YY_rt, xx_rt, yy_rt = return_time_fix(A_max_sorted)
    print(xx_rt,yy_rt)

    tau = np.arange(-30,30,1)
    nb_events = 0
    myfield.composite_mean = np.zeros((len(tau),myfield.var.shape[2],myfield.var.shape[3]))
    myfield.composite_std = np.zeros((len(tau),myfield.var.shape[2],myfield.var.shape[3]))
    for y in range(series.shape[0]):
        if yy_rt[return_index[0]] <= A_max[y] < yy_rt[return_index[1]]:
            print("A_max["+str(y)+"] = ",A_max[y], "Ti["+str(y)+"] = ", Ti[y])
            nb_events += 1
            value = (myfield.var[y])[tau + (Tot_Mon1[start_month] + Ti[y]) ]
            myfield.composite_mean += value     # This is the raw sum
            myfield.composite_std += value**2   # This is the raw square sum
    print("number of events: ",nb_events)
    # std and mean are computed below
    myfield.composite_std = np.sqrt((myfield.composite_std - (myfield.composite_mean * myfield.composite_mean / nb_events)) / (nb_events - 1))
    myfield.composite_mean /= nb_events
    myfield.composite_t = (lambda a, b: np.divide(a, b, out=np.zeros(a.shape), where=b != 0))(np.sqrt(nb_events) * myfield.composite_mean, myfield.composite_std)

def geo_contour(m, ax, Center_map, Lon, Lat, data_contour_value, data_contour_level, colmap1, colmap2):
    '''
    Plots a contour using two different colormaps for positive and negative anomalies
    
    ax, Center_map, aren't used
    '''
    if plotter == 'cartopy':
        return cplt.geo_contour(m, Lon, Lat, data_contour_value,
                                levels=data_contour_level, cmap1=colmap1, cmap2=colmap2)
    
    c_sign = m.contour(Lon, Lat, data_contour_value,
                       levels=data_contour_level, cmap=colmap1,linewidths=1, linestyles="dashed",
                       latlon=True, vmin=data_contour_level[0], vmax=0)
    subset = data_contour_value.copy()
    subset[subset<0] = 0 
    c_sign = m.contour(Lon, Lat, subset,
                       levels=data_contour_level, cmap=colmap2,linewidths=1,
                       latlon=True, vmin=0, vmax=data_contour_level[-1])
    
    # this is confusing
    fmt = '%1.0f'
    v_sign = data_contour_level[int(len(data_contour_level) / 2)-1], data_contour_level[int(len(data_contour_level) / 2)]
    if len(c_sign.levels) > len(v_sign): # len(v_sign) = 2 because it is a 2-uple
        p.clabel(c_sign, v_sign, inline=True,fmt=fmt,fontsize=14)

def geo_contourf(m, ax, Center_map, Lon, Lat, data_colorbar_value, data_colorbar_level, colmap, title_frame, put_colorbar=True, draw_gridlines=True):
    '''
    ax, Center_Map aren't used
    '''
    if plotter == 'cartopy':
        return cplt.geo_contourf(m, Lon, Lat, data_colorbar_value,
                                 levels=data_colorbar_level, cmap=colmap, title=title_frame, put_colorbar=put_colorbar, draw_gridlines=draw_gridlines)
    
    plt.cla()
    m.contourf(Lon, Lat, data_colorbar_value, levels=data_colorbar_level, cmap=colmap, extend='both', latlon=True)
    if put_colorbar:
        m.colorbar()
    m.drawcoastlines(color='black',linewidth=1)
    m.drawparallels(np.arange(-80.,81.,20.),linewidth=0.5,labels=[True,False,False,False], color = "green")
    m.drawmeridians(np.arange(-180.,181.,20.),linewidth=0.5,labels=[False,False,False,True],color = "green")
    plt.title(title_frame, fontsize=20)
    
def geo_contour_color(m, ax, Center_map, Lon, Lat, T_value, data_contour_value, data_contour_t, data_contour_level, colors, mylinestyles, mylinewidths):
    '''
    ax, Center_map not used
    '''
    fmt = '%1.0f'
    fontsize = 12
    
    if plotter == 'cartopy':
        return cplt.geo_contour_color(m, Lon, Lat, data_contour_value, data_contour_t, T_value,
                                      levels=data_contour_level, colors=colors, linestyles=mylinestyles,
                                      linewidths=mylinewidths, fmt=fmt, fontsize=fontsize)
    
    zg_sign, zg_not, zg_taken = significative_data2(data_contour_value, data_contour_t, T_value, True)
    c_nots = m.contour(Lon, Lat, data_contour_value, levels=data_contour_level[:data_contour_level.shape[0]//2], colors=colors[1], linestyles = mylinestyles[1], linewidths=mylinewidths[1], latlon=True) #negative insignificant anomalies of geopotential
    v_sign = data_contour_level[int(len(data_contour_level) / 2)-1], # data_contour_level[int(len(data_contour_level) / 2)]
    if len(c_nots.levels) > len(v_sign):
        p.clabel(c_nots, v_sign, inline=True,fmt=fmt,fontsize=fontsize)
    c_nots = m.contour(Lon, Lat, data_contour_value, levels=data_contour_level[data_contour_level.shape[0]//2:], colors=colors[2], linestyles = mylinestyles[2],linewidths=mylinewidths[2], latlon=True)  #positive insignificant anomalies of geopotential
    v_sign = data_contour_level[int(len(data_contour_level) / 2)],
    if len(c_nots.levels) > len(v_sign):
        p.clabel(c_nots, v_sign, inline=True,fmt=fmt,fontsize=fontsize)
    c_sign = m.contour(Lon, Lat, zg_sign, levels=data_contour_level[:data_contour_level.shape[0]//2], colors=colors[0], linestyles = mylinestyles[0],linewidths=mylinewidths[0], latlon=True)  #negative significant anomalies of geopotential
    c_sign = m.contour(Lon, Lat, zg_sign, levels=data_contour_level[data_contour_level.shape[0]//2:], colors=colors[3], linestyles = mylinestyles[3],linewidths=mylinewidths[3], latlon=True)   #positive significant anomalies of geopotential
    
def PltMaxMinValue(m,Lon, Lat, data_contour_value):
    if plotter == 'cartopy':
        return cplt.PltMaxMinValue(m, Lon, Lat, data_contour_value)
        
    coordsmax = np.unravel_index(np.argmin(data_contour_value, axis=None), data_contour_value.shape)
    x, y = m(Lon[coordsmax[0], coordsmax[1]], Lat[coordsmax[0], coordsmax[1]])
    txt = plt.text(x, y, "{:1.0f}".format(np.min(data_contour_value)), color='red')
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
    
    coordsmax = np.unravel_index(np.argmax(data_contour_value, axis=None), data_contour_value.shape)
    x, y = m(Lon[coordsmax[0], coordsmax[1]], Lat[coordsmax[0], coordsmax[1]])
    txt = plt.text(x, y, "{:1.0f}".format(np.max(data_contour_value)), color='blue')
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])


        
def anomaly_animate(m, ax, Center_map, Lon, Lat, data_colorbar_value, data_colorbar_level,
            data_contour_value, data_contour_level, colmap, title_frame):
    if plotter == 'cartopy':
        raise NotImplementedError("Use cartopy_plots.animate")
    fmt = '%1.0f'
    plt.cla()
    m.contourf(Lon, Lat, data_colorbar_value, levels=data_colorbar_level, cmap=colmap, extend='both', latlon=True)
    m.colorbar()
    c_sign = m.contour(Lon, Lat, data_contour_value, levels=data_contour_level, cmap="PuRd",linewidths=1, linestyles = "dashed", latlon=True, vmin = data_contour_level[0], vmax = 0)
    subset = data_contour_value.copy()
    print(subset.shape)
    coordsmax = np.unravel_index(np.argmin(data_contour_value[:32,:], axis=None), data_contour_value[:32,:].shape)
    x, y = m(Lon[coordsmax[0], coordsmax[1]], Lat[coordsmax[0], coordsmax[1]])
    txt = plt.text(x, y, "{:1.0f}".format(np.min(data_contour_value[:32,:])), color='red')
    subset[subset<0] = 0 
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
    print(subset.shape)
    c_sign = m.contour(Lon, Lat, subset, levels=data_contour_level, cmap="summer",linewidths=1, latlon=True, vmin = 0, vmax = data_contour_level[-1])
    v_sign = data_contour_level[int(len(data_contour_level) / 2)-1], data_contour_level[int(len(data_contour_level) / 2)]
    coordsmax = np.unravel_index(np.argmax(data_contour_value[:32,:], axis=None), data_contour_value[:32,:].shape)
    x, y = m(Lon[coordsmax[0], coordsmax[1]], Lat[coordsmax[0], coordsmax[1]])
    txt = plt.text(x, y, "{:1.0f}".format(np.max(data_contour_value[:32,:])), color='blue')
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
    if len(c_sign.levels) > len(v_sign):
        p.clabel(c_sign, v_sign, inline=True,fmt =fmt,fontsize=14)
    m.drawcoastlines(color='black',linewidth=1.5)
    m.drawparallels(np.arange(-80.,81.,20.),linewidth=0.75,labels=[True,False,False,False], color = "black")
    m.drawmeridians(np.arange(-180.,181.,20.),linewidth=0.75,labels=[False,False,False,True],color = "black")
    plt.title(title_frame, fontsize=20)
    
def absolute_animate(i, m, ax, Center_map, Nb_frame, Lon, Lat, data_colorbar_value, data_colorbar_level,
            data_contour_value, data_contour_level, colmap, title_frame):
    if plotter == 'cartopy':
        raise NotImplementedError("Use cartopy_plots.animate")
    fmt = '%1.0f'
    print('i:', i)
    plt.cla()
    
    coordsmax = np.unravel_index(np.argmax(data_contour_value[:Lon.shape[0]//2,:], axis=None), data_contour_value[:Lat.shape[0]//2,:].shape)
    x, y = m(Lon[coordsmax[0], coordsmax[1]], Lat[coordsmax[0], coordsmax[1]])
    maximum = np.max(data_contour_value[:Lat.shape[0]//2,:])
    txt = plt.text(x, y, "{:1.0f}".format(maximum), color='blue')
    
    coordsmax2 = np.unravel_index(np.argmin(data_contour_value[:Lon.shape[0]//2,:], axis=None), data_contour_value[:Lat.shape[0]//2,:].shape)
    x2, y2 = m(Lon[coordsmax2[0], coordsmax2[1]], Lat[coordsmax2[0], coordsmax2[1]])
    minimum = np.min(data_contour_value[:Lat.shape[0]//2,:])
    txt2 = plt.text(x2, y2, "{:1.0f}".format(minimum), color='red')
    
    
    c_sign_f = m.contourf(Lon, Lat, data_contour_value, levels=data_contour_level, latlon=True, cmap = "GnBu", vmin = data_contour_level[15], vmax = data_contour_level[-20])#colors="lime")
    cbar=m.colorbar(location="bottom")
    contf = m.contourf(Lon, Lat, data_colorbar_value, levels=data_colorbar_level, cmap=colmap, extend='both', latlon=True)
    m.colorbar()
    
    c_sign = m.contour(Lon, Lat, data_contour_value, levels=data_contour_level,linewidths=2, latlon=True, cmap = "GnBu", vmin = data_contour_level[15], vmax = data_contour_level[-20])#colors="lime")
    v_sign = data_contour_level[int(len(data_contour_level) / 2)-1], data_contour_level[int(len(data_contour_level) / 2)]

    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
    txt2.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
    m.drawcoastlines(color='pink',linewidth=1)
    m.drawparallels(np.arange(-80.,81.,20.),linewidth=0.75,labels=[True,False,False,False], color = "white")
    m.drawmeridians(np.arange(-180.,181.,20.),linewidth=0.75,labels=[False,False,False,True],color = "white")
    plt.title(title_frame)
    
def anomaly_absolute_animate(m, ax, Center_map, Lon, Lat, data_colorbar_value, data_colorbar_level,
            data_contour_value, data_contour_level, colmap, title_frame):
    if plotter == 'cartopy':
        raise NotImplementedError("Use cartopy_plots.animate")
    fmt = '%1.0f'
    plt.cla()
    m.contourf(Lon, Lat, data_colorbar_value, levels=data_colorbar_level, cmap=colmap, extend='both', latlon=True)
    m.colorbar()
    CS = m.contour(Lon, Lat, data_contour_value, levels=data_contour_level, colors='darkgreen',linewidths=2, latlon=True)
    ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)
    m.drawcoastlines(color='black',linewidth=1.5)
    m.drawparallels(np.arange(-80.,81.,20.),linewidth=0.75,labels=[True,False,False,False], color = "black")
    m.drawmeridians(np.arange(-180.,181.,20.),linewidth=0.75,labels=[False,False,False,True],color = "black")
    ax.set_title(title_frame)
    
def full_extent(ax, padx=0.0, pady=[]):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    if pady == []:
        pady = padx
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = []
    if ax.get_xscale() == 'log':
        items += ax.get_xticklabels()[2:-2] # weird logscale behavior
    else:
        items += ax.get_xticklabels()[1:-1]
    if ax.get_yscale() == 'log':
        items += ax.get_yticklabels()[-2:2]
    else:
        items += ax.get_yticklabels()[-1:1]
    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
#    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + padx, 1.0 + pady)

    
def return_time_fix(D_sorted, modified='no', specific_returns=[1, 4, 10, 40, 100, 1000]): # In this function we fix the length
    '''
    Computes the return time from    
    D_sorted: sorted dictionary with layout {anomaly: [day, year]}
    specific_returns: list
        list of return times that will populate x_rt and y_rt    
        
    If modified == 'no':
        the return time `tau` for anomaly `a` is
        
        `tau` = `N`/`m`
        
        Where `N` is the total number of events in D_sorted and `m` is the number of events which have an anomaly >= `a`
        
    If modified == 'yes':
        
        `tau` = 1/log(1 - `m`/`N`)
        
        For an explanation of this formula look at
        T. Lestang et al. 'Computing return times or return periods with rare event algorithms'.
        DOI: https://doi.org/10.1088/1742-5468/aab856
    '''
    m = 1 # index that counts how many extreme events are more extreme than a given one. Since the events are ordered, this is just the event counter.
    Y_rt = []
    X_rt = []
    y_rt = []
    x_rt = []
    for key in D_sorted:
        if modified == 'yes':
            r1 = - 1/(np.log(1 - m/len(D_sorted))) # assumption of Poisson process
        else:
            r1 = len(D_sorted) / m  # compute return time
        Y_rt.append(key[0])
        X_rt.append(r1)
        m += 1
        
    for r1s in [1, 4, 10, 40, 100, 1000]:
        r1s_closest = min(X_rt, key=lambda x:abs(x-r1s))
        x_rt.append(r1s_closest)
        y_rt.append(Y_rt[X_rt.index(r1s_closest)])    
    return X_rt, Y_rt, x_rt, y_rt

def a_max_and_ti_postproc(A, length=None):
    """
    Generates unranked set of maximal anomalies per each year and when they occur.
    `A` needs to have an extra point at the beginning and end of each year to check if a maximum at the extremes is local or not.    
    """
    # Probably outdated comment
    # In the code A is expected to be loaded from June1 - 1   to August16 + 1 to check the maxima at the boundaries
    
    just_max_index = []
    if length is None:
        A_summer = A[ :, 1:-1]  # allow to check maxima at the edges
        length = A_summer.shape[1]
    else:
        A_summer = A[:, 1:length+1]
        if A_summer.shape[1] < length:
            warnings.warn('a_max_and_ti_postproc: adjusting length')
            length = A_summer.shape[1]
    #print('    verif: we look A(t) over {} index (excepted value={})'.format(len(A_summer[0]), length))
    out_A_max = []
    out_Ti = []
    out_year_a = []
    year_with_before_non_loc = []  # this way we have data about the maxima (where they belong)
    year_with_after_non_loc = []
    year_with_before = []
    year_with_after = []
    start_true = 0
    end_true = 0
    for j in range(len(A)):  # j=year
        max_index = np.argmax(A_summer[j])  # the time during season when we have a maximum this year
        max_value = A_summer[j][max_index]  # the corresponding value of the maximum of A
        just_max_index.append(max_index)  # collect t_i
        logger.debug(f"{max_index = } is compared to {length - 1}")
        if max_index == 0:
            if A[j][0] > max_value:  # check if the maximum is a false maximum
                a_max, ti = maximum_inside(A_summer[j]) # find another true maximum that is a local maximum inside
                year_with_before_non_loc.append(j)  # provide the corresponding year
                #print("year ",j," start rejected")
            else:
                #print("year ",j," start accepted")
                a_max = max_value  # if it is a true maximum keep it
                ti = max_index
                year_with_before.append(j)
                start_true += 1
        elif max_index == length - 1:  # do the same on the other side
            logger.debug(f"{max_index = } triggered")
            if A[j][-1] > max_value:
                #print("year ",j," end rejected")
                a_max, ti = maximum_inside(A_summer[j])
                #print("New ti = ", ti)
                year_with_after_non_loc.append(j)
            else:
                #print("year ",j," end accepted")
                a_max = max_value
                ti = max_index
                year_with_after.append(j)
                end_true += 1
        else:
            a_max = max_value  # if we are not at the boundary do the standard maximum extraction
            ti = max_index
        out_A_max.append(a_max)
        out_Ti.append(ti + 1) # shift ti values by one to compensate the fact that we ignored the first element of A
        out_year_a.append(j)  # the year is somewhat redundant as it is always the same set of years
    #print('    there are {} years where the maximum is not the np.max maximum'.format(
    #    len(year_with_after_non_loc) + len(year_with_before_non_loc)))
    #print("start_true = ", start_true, ", end_true", end_true)    
    return out_A_max, out_Ti, out_year_a

def maximum_inside(data):
    '''
    Finds the maximum of local maxima in a 1D array
    
    Returns the maximum value and its index in the array
    '''
    local_max_indexs = argrelextrema(data, np.greater)  # find local maxima
    ti_list = local_max_indexs[0]
    a_max_list = data[ti_list]
    if len(a_max_list) == 0: # no local maxima
        index = np.argmax(data)
        maximum = data[index]
    else:
        # maximum of local maxima
        index = ti_list[np.argmax(a_max_list)]
        maximum = data[index]
    return maximum, index

def a_decrese(in_A_max, in_Ti, in_year_a):
    """
    Creates a table for the ranked extreme heatwaves: threshold in_A_max,
    Time during a season in_Ti and the corresponding year in_year_a
    """
    D = {}
    if len(in_A_max) == len(in_Ti) and len(in_A_max) == len(in_year_a):
        for i in range(len(in_A_max)):
            D[i] = [in_Ti[i], in_year_a[i],in_A_max[i]] 
    else: # In this version of the code we avoid shorter sequences when there are dublicated of int_A_max
        logger.warning(f'size mismatch: {len(in_A_max) = },{len(in_Ti) = },{len(in_year_a) = }')
    D_sorted = sorted(D.items(), key=lambda kv: kv[1][2], reverse=True)
    return [(D_sorted_i[1][2],D_sorted_i[1][:2]) for D_sorted_i in D_sorted]


def draw_map(m, scale=0.2, background='stock_img', **kwargs):
    '''
    Plots a background map.
    
    If plotting with basemap additional parameters are ignored
    
    If plotting with cartopy `scale` is ignored
    '''
    if plotter == 'cartopy':
        return cplt.draw_map(m, background, **kwargs)
    # draw a shaded-relief image
    m.shadedrelief(scale=scale)
    
    # lats and longs are returned as a dictionary
    lats = m.drawparallels(np.linspace(-90, 90, 13))
    lons = m.drawmeridians(np.linspace(-180, 180, 13))

    # keys contain the plt.Line2D instances
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)
    
    # cycle through these lines and set the desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')

def is_above_line(da:xr.DataArray, lon1:float, lat1:float, lon2:float, lat2:float) -> xr.DataArray:
    '''
    returns a mask of the input object that is true north of a line in lon-lat space defined by two points.

    By multiplying the output of this funcion over several evaluations you can define a polygonal mask

    Parameters
    ----------
    da : xr.DataArray
        input object with longitude and latitude dimensions
    lon1 : float
        longitude of the first point
    lat1 : float
        latitude of the first point
    lon2 : float
        longitude of the second point
    lat2 : float
        latitude of the second point

    Returns
    -------
    xr.DataArray
        the mask
    '''
    da = standardize_dim_names(da)
    return da.lat - (lat1*(lon2 - da.lon) + lat2*(da.lon - lon1))/(lon2 - lon1) > 0

def create_mask_xarray(model:str, area:str, lsm:xr.DataArray) -> xr.DataArray:
    '''
    Returns a boolean mask that is true over the area of interest, using xarray features.
    It is similar to `create_mask` with the option return_full_mask = True.
    However this time the mask is created using explicit latitude and longitude values, allowing also non square masks

    Parameters
    ----------
    model : str
        name of the model, e.g. 'Plasim', 'CESM', 'ERA5'
    area : str
        name of the area to mask, e.g. France
    lsm : xr.DataArray
        land-sea mask for the model, having only longitude and latitude dimensions

    Returns
    -------
    xr.DataArray
        the mask, same shape as `lsm`

    Raises
    ------
    NotImplementedError
        If the combination model/area is not implemented
    '''
    if model == 'ERA5':
        if area == 'France':
            mask = standardize_dim_names(lsm > 0.5) # convert to bool keeping only the land masses
            mask *= (mask.lat < 52)*(mask.lat > 42)*(mask.lon > -5)*(mask.lon < 8.3) # identify the rough region
            mask *= ~is_above_line(mask, 1.65, 51, -4.5, 49.2)
            mask *= is_above_line(mask, -1.86, 43.34, 3.4, 42.2)
            mask *= ~is_above_line(mask, 2.26, 51.2, 8.27, 49)
            mask *= is_above_line(mask, 8.1, 48.8, 6, 43)
            return mask
        
    raise NotImplementedError(f'{model}:{area}')

# now vectorized :)
def create_mask(model:str, area:str, data:np.ndarray, axes='first 2', return_full_mask=False): # careful, this mask works if we load the full Earth. there might be problems if we extract fields from some edge of the map
    """
    This function allows to extract a subset of data enclosed in the area.
    The output has the dimension of the area on the axes corresponding to latitued and longitude
    If the area includes the Greenwich meridian, a concatenation is required.
    
    If `axes` == 'last 2', the slicing will be done on the last 2 axes,
    otherwise on the first two axes
    
    If `return_full_mask` == True, the function returns an array with the same shape of `data`, True over the `area` and False elsewhere
    Otherwise the return will be `data` restricted to `area`
    """
    
    if axes == 'first 2' and len(data.shape) > 2:
        # permute the shape so that the first 2 axes end up being the last 2
        _data = data.transpose(*range(2,len(data.shape)),0,1)
        _data = create_mask(model, area, _data, axes='last 2', return_full_mask=return_full_mask)
        # permute the axes back to their original condition
        return _data.transpose(-2, -1, *range(len(data.shape) - 2))
    
    if return_full_mask:
        mask = np.zeros_like(data, dtype=bool)
    
    if model == 'ERA5':
        logger.warning('Creating mask on ERA data with old code!')
        if area == "Scandinavia":
            if return_full_mask:
                mask[...,25:45,7:53] = True
                return mask
            return data[...,25:45,7:53]# reconstructed from Francesco
        elif area == "Scand": # = Norway Sweden
            if return_full_mask:
                mask[...,25:45,7:30] = True
                return mask
            return data[...,25:45,7:30]
        elif area == "NAtlantic":
            if return_full_mask:
                mask[...,25:80, -135:] = True
                mask[...,25:80, :60] = True
                return mask
            return np.concatenate((data[...,25:80, -135:], data[...,25:80, :60]), axis=-1)  # used for plotting
        elif area == "France":
            if return_full_mask:
                mask[...,51:63, -4:] = True
                mask[...,51:63, :9] = True
                return mask
            return np.concatenate((data[...,51:63, -4:], data[...,51:63, :9]), axis=-1)  # reconstructed from Francesco
        elif area == "France_bis":
            if return_full_mask:
                mask[...,52:64, -5:] = True
                mask[...,52:64, :9] = True
                return mask
            return np.concatenate((data[...,52:64, -5:], data[...,52:64, :9]), axis=-1)  # fixing to CESM
        elif area == "Russia":  # lat[i]<60 and lat[i]>50: index 9-15
            if return_full_mask:
                mask[...,37:60, 42:79] = True
                return mask
            return data[...,37:60, 42:79] 
        elif area == "Poland":  # From stefanon
            if return_full_mask:
                mask[...,44:60, 18:43] = True
                return mask
            return data[...,44:60, 18:43]
        else:
            logger.error(f'Unknown area {area}')
            return None
    
    elif model == "CESM":
        if area == "France":
            if return_full_mask:
                mask[...,-51:-41, -3:] = True
                mask[...,-51:-41, :6] = True
                return mask
            return np.concatenate((data[...,-51:-41, -3:],data[...,-51:-41, :6]), axis=-1)
        elif area == "Scandinavia":
            if return_full_mask:
                mask[...,-36:-20, 4:32] = True
                return mask
            return data[...,-36:-20, 4:32]
        elif area == "Scand": # = Norway Sweden
            if return_full_mask:
                mask[...,-36:-20, 4:18] = True
                return mask
            return data[...,-36:-20, 4:18]
        elif area == "Russia":  # lat[i]<60 and lat[i]>50: index 9-15
            if return_full_mask:
                mask[...,-48:-29, 25:48] = True
                return mask
            return data[...,-48:-29, 25:48]  # lon[i]<55 and lon[i]>35:   index 11-21
        elif area == "Poland":  # From Stefanon
            if return_full_mask:
                mask[...,-48:-35, 11:26] = True
                return mask
            return data[...,-48:-35, 11:26]
        else:
            logger.error(f'Unknown area {area}')
            return None
    
    elif model == "Plasim":
        if area == "NW_Europe":
            if return_full_mask:
                mask[...,10:16, -1:] = True
                mask[...,10:16, :7] = True
                return mask
            return np.concatenate((data[...,10:16, -1:], data[...,10:16, :7]), axis=-1)  # give by Valerian/Francesco 
        elif area == "Greenland":
            if return_full_mask:
                mask[...,2:9, 108:120] = True
                return mask
            return data[...,2:9, 108:120]
        elif area == "Europe":
            if return_full_mask:
                mask[...,7:19, -3:] = True
                mask[...,7:19, :10] = True
                return mask
            return np.concatenate((data[...,7:19, -3:], data[...,7:19, :10]), axis=-1)  # give by Valerian/Francesco  
        elif area == "France":
            if return_full_mask:
                mask[...,13:17, -1:] = True
                mask[...,13:17, :3] = True
                return mask
            return np.concatenate((data[...,13:17, -1:], data[...,13:17, :3]), axis=-1)  # give by Valerian
        elif area == "Quebec":  # lat[i]<60 and lat[i]>50:      index: 10-13
            if return_full_mask:
                mask[...,10:16, 98:110] = True
                return mask
            return data[...,10:16, 98:110]  # lon[i]<180+120 and lon[i]>180+110   index:104-106
        elif area == "USA":  # lat[i]<50 and lat[i]>25:  index: 14-22
            if return_full_mask:
                mask[...,14:23, 89:109] = True
                return mask
            return data[...,14:23, 89:109]  # lon[i]<180+125 and lon[i]>180+70:  index 89-108
        elif area == "US":  # lat[i]<50 and lat[i]>25:  index: 14-22   # < fixing the area of philipinne
            if return_full_mask:
                mask[...,14:23, 84:104] = True
                return mask
            return data[...,14:23, 84:104]  # lon[i]<180+125 and lon[i]>180+70:  index 89-108
        elif area == "Midwest":
            if return_full_mask:
                mask[...,16:20, 92:99] = True
                return mask
            return data[...,16:20, 92:99]
        elif area == "Alberta":
            if return_full_mask:
                mask[...,10:15, 85:90] = True
                return mask
            return data[...,10:15, 85:90]
        elif area == "Scandinavia":  # 55<lat<72: index: 6-11
            if return_full_mask:
                mask[...,6:12, 2:15] = True
                return mask
            return data[...,6:12, 2:15]  # 5<lon<40 : index 2-14
        elif area == "Scand":  #  = Norway Sweden
            if return_full_mask:
                mask[...,6:12, 2:8] = True
                return mask
            return data[...,6:12, 2:8]  # 5<lon<40 : index 2-14
        elif area == "Russia":  # lat[i]<60 and lat[i]>50: index 9-15
            if return_full_mask:
                mask[...,9:16, 11:22] = True
                return mask
            return data[...,9:16, 11:22]  # lon[i]<55 and lon[i]>35:   index 11-21
        elif area == "Poland":  # lat[i]<60 and lat[i]>50: index 9-15
            if return_full_mask:
                mask[...,11:16, 5:12] = True
                return mask
            return data[...,11:16, 5:12]  # lon[i]<55 and lon[i]>35:   index 11-21
        elif area == 'total':  # return all data, use for total_area function and create surface over continents
            if return_full_mask:
                return np.ones_like(data, dtype=bool)
            return data
        else:
            logger.error(f'Unknown area {area}')
            return None
    else:
        logger.error(f'Unknown model {model}')
        return None

def Greenwich(Myarray):
    '''
    Returns `Myarray` with the Greenwich meridian counted twice (start and end of the array): useful for plotting
    '''
    # old
    # return np.append(Myarray,np.transpose([Myarray[:,0]]),axis = 1)
    return np.append(Myarray, Myarray[...,0:1], axis=-1) # this works for an array of arbitrary shape where the last index is longitude. using 0:1 instead of just 0 allows to have the correct number of dimensions
    
def autocorrelation(myseries, maxlag):
    series_pad = np.pad(myseries,((0, 0), (0, maxlag)), 'constant')  # this pads each year with padsize sample time of 0s so that when the array is permuted to be multiplied by itself we don't end up using the previous part of the year
    autocorr = []
    for k in range(maxlag):
        autocorr.append(np.sum(series_pad*np.roll(series_pad, -k))/(series_pad.shape[0]*(series_pad.shape[1]-k-maxlag)))
    return autocorr



class Field:
    def __init__(self, name, filename, label, Model, shape, s_year, prefix,
                 day_start=0, day_end=365, lat_start=0, lat_end=241, lon_start=0, lon_end=480, myprecision='double'):
        if myprecision == 'double':
            self.np_precision = np.float64
            self.np_precision_complex = np.complex128
        else: # we can save space since for learning we don't need as much precision
            self.np_precision = np.float32
            self.np_precision_complex = np.complex64
        self.start_year = s_year # The first year in the database
        self.name = name    # Name inside the .nc file
        self.filename = filename # Name of the .nc file 
        self.label = label  # Label to be displayed on the graph
        self.var = np.zeros(shape, dtype=self.np_precision)        # The actual field data
        self.time = np.zeros((shape[0],shape[1]), dtype=self.np_precision)        # local time in days
        self.abs_mask = np.zeros((shape[0],shape[1]), dtype=self.np_precision)    # integral over the area
        
        self.detr_mask = np.zeros((shape[0],shape[1]), dtype=self.np_precision)    # integral over the area of the detrended field
        
        self.ano_mask = np.mean((shape[0],shape[1]), dtype=self.np_precision)      # integral over the area of the anomaly of the field
        
        self.coef = np.zeros((shape[2],shape[3]), dtype=self.np_precision)        # used for detrending
        self.intercept = np.zeros((shape[2],shape[3]), dtype=self.np_precision)   # used for detrending
        self.fit_time = np.zeros((shape[0],1), dtype=self.np_precision)           # used for keeping track of a time in fraction of a year centered on the summer
        self.detr_mean = np.zeros((shape[1],shape[2],shape[3]), dtype=self.np_precision)        # detrended climatological mean
        self.detr_std = np.zeros((shape[1],shape[2],shape[3]), dtype=self.np_precision)        # detrended climatological mean
        self.prefix = prefix                                            # a prefix of the data input name
        self.day_start = day_start   # below we put bounds on the input to save RAM
        self.day_end = day_end       
        self.lat_start = lat_start
        self.lat_end = lat_end
        self.lon_start = lon_start
        self.lon_end = lon_end
        self.Model = Model
        
        
    def add_year(self, year):   # Load the year from the database
        dataset = Dataset('Data_ERA5/'+self.prefix+'ERA5_'+self.filename+'_'+str(year+self.start_year)+'.nc')
        logger.info(f"Loading field {self.name}, {year = }, dataset.variables[{self.name}].shape = {dataset.variables[self.name].shape}")
        if ((year + self.start_year)%4): # if not divisible by 4 it is not a leap year
            if len(dataset.variables[self.name].shape) < 4: 
                self.var[year] = np.asarray(dataset.variables[self.name][self.day_start:self.day_end,self.lat_start:self.lat_end,self.lon_start:self.lon_end], dtype=self.np_precision)
            else: # for some reason year 2020 has two layers
                self.var[year] = np.asarray(dataset.variables[self.name][self.day_start:self.day_end,0,self.lat_start:self.lat_end,self.lon_start:self.lon_end], dtype=self.np_precision)
            self.time[year] = np.asarray(dataset.variables['time'][self.day_start:self.day_end], dtype=self.np_precision)/24  # Convert from hours to days
        else:  # This is a leap year so we are going to subtract one day from the start of the year to match the 365 length
            if len(dataset.variables[self.name].shape) < 4:
                self.var[year] = np.asarray(dataset.variables[self.name][1:], dtype=self.np_precision)[self.day_start:self.day_end,self.lat_start:self.lat_end,self.lon_start:self.lon_end]
            else:
                self.var[year] = np.asarray(dataset.variables[self.name][1:], dtype=self.np_precision)[self.day_start:self.day_end,0,self.lat_start:self.lat_end,self.lon_start:self.lon_end]
            self.time[year] = np.asarray(dataset.variables['time'][1:], dtype=self.np_precision)[self.day_start:self.day_end]/24  # Convert from hours to days
        dataset.close()
        
    def Greenwich(self, year, day):  # Take an array and copy the Greenwhich meridian to make sure we avoid a segment hole in basemap
        '''
        Returns `self.var` with the Greenwich meridian counted twice (start and end of the array): useful for plotting
        
        Parameters:
        year: int or 'all' (this last one only if `day` == 'all')
        day: int or 'all'
        '''
        # old
        # return np.append(self.var[year,day,:,:],np.transpose([self.var[year,day,:,0]]),axis = 1)
        if day == 'all':
            if year == 'all':
                return Greenwich(self.var)
            return Greenwich(self.var[year])
        if year == 'all':
            raise NotImplementedError('if you specify a day, you must also specify a year')
        return Greenwich(self.var[year,day])
    
    
    def Set_area_integral(self, input_area, input_mask):   # Evaluate area integral
        series = np.zeros((self.var.shape[0],self.var.shape[1]), dtype=self.np_precision)
        # TO BE UPDATED
        for y in range(self.var.shape[0]):
            for i in range(self.var.shape[1]):
                self.abs_mask[y,i] = np.sum(np.sum(create_mask(self.Model,input_area,self.var[y,i,:,:])*input_mask))
        self.ano_abs_mask = self.abs_mask - np.mean(self.abs_mask,0)    # Evaluate grid-point climatological mean 
        anomaly_series = self.ano_abs_mask.copy()
        series = self.abs_mask.copy()
        return series, anomaly_series
                
    def Set_Detrend(self,period, lat, lon, FitKind):  # Detrend the climate change in the part of the year defined via period. 
        # Detrend each grid point. 
        
        self.fit_time = ((self.time - self.time[0,0]-(period[0]+period[1])/2)/365)   # Convert day of a year into a fraction of a year, normalized to its middle
        self.detrended = np.zeros(self.var.shape, dtype=self.np_precision)          # Detrended field 
        if os.path.exists('ERA5/'+FitKind+'/'+self.prefix+'coef_intercept'+self.name+'.nc'):
            ##### LOAD ####
            dataset = Dataset('ERA5/'+FitKind+'/'+self.prefix+'coef_intercept'+self.name+'.nc')
            self.coef = np.asarray(dataset.variables[self.name+'_coef'][:], dtype=self.np_precision)[self.lat_start:self.lat_end,self.lon_start:self.lon_end]
            self.intercept = np.asarray(dataset.variables[self.name][:], dtype=self.np_precision)[self.lat_start:self.lat_end,self.lon_start:self.lon_end]
            dataset.close()
            for y in range(self.var.shape[0]):
                logger.debug(f"loading year {y}")
                dataset = Dataset('ERA5/'+FitKind+'/'+self.prefix+self.name+'_detrended_fields_'+str(self.start_year+y)+'.nc')
                self.detrended[y] = np.asarray(dataset.variables[self.name+'_det'][:,self.lat_start:self.lat_end,self.lon_start:self.lon_end], dtype=self.np_precision)
                dataset.close()
        else: # if the file doesn't exist we must peform detrending
            summer_mean = np.mean(self.var[:,period[0]:period[1],:,:],1) # yearly mean over each summer defined by the period[0] - period[1]
            X =  np.mean(self.fit_time[:,period[0]:period[1]],1).reshape((-1, 1))    # corresponding time
            for lat_loop in range(self.coef.shape[0]):
                logger.debug(f"latitude index {lat_loop}")
                for lon_loop in range(self.coef.shape[1]):  # perform a linear fit for each grid point
                    if FitKind == 'linear': 
                        model = LinearRegression().fit(X, summer_mean[:,lat_loop,lon_loop])
                        self.intercept[lat_loop,lon_loop] = model.intercept_
                        self.coef[lat_loop,lon_loop] = model.coef_
                        self.detrended[:,:,lat_loop,lon_loop] = self.var[:,:,lat_loop,lon_loop] - model.predict(self.fit_time.reshape((-1, 1))).reshape(self.time.shape)  # this is where detrending happens
                    else:
                        poly = PolynomialFeatures(degree = 2) #quadratic
                        X_poly = poly.fit_transform(X)
                        poly.fit(X_poly, summer_mean[:,lat_loop,lon_loop])
                        lin2 = LinearRegression()
                        lin2.fit(X_poly, summer_mean[:,lat_loop,lon_loop])
                        self.intercept[lat_loop,lon_loop] = lin2.intercept_
                        self.coef[lat_loop,lon_loop] = lin2.coef_[1]
                        self.detrended[:,:,lat_loop,lon_loop] = self.var[:,:,lat_loop,lon_loop] - lin2.predict(poly.fit_transform(self.fit_time.reshape((-1, 1)))).reshape(self.time.shape)
   
            ###### SAVE ########
            ncfile = Dataset('ERA5/'+FitKind+'/'+self.prefix+'coef_intercept'+self.name+'.nc',mode='w',format='NETCDF4_CLASSIC') 
            lat_dim = ncfile.createDimension('lat', len(lat))     # latitude axis
            lon_dim = ncfile.createDimension('lon', len(lon))    # longitude axis
            lat_det = ncfile.createVariable('lat', np.float32, ('lat',))
            lat_det.units = 'degrees_north'
            lat_det.long_name = 'latitude'
            lon_det = ncfile.createVariable('lon', np.float32, ('lon',))
            lon_det.units = 'degrees_east'
            lon_det.long_name = 'longitude'
            # Define a 3D variable to hold the data
            NCcoef = ncfile.createVariable(self.name+'_coef',self.np_precision,('lat','lon')) # note: unlimited dimension is leftmost
            NCcoef.standard_name = self.label+' coefficient' # this is a CF standard name
            NCinter = ncfile.createVariable(self.name,self.np_precision,('lat','lon')) # note: unlimited dimension is leftmost
            NCinter.standard_name = self.label+' intercept' # this is a CF standard name

            lat_det[:] = lat
            lon_det[:] = lon
            NCcoef[:,:] = self.coef
            NCinter[:,:] = self.intercept
            # first print the Dataset object to see what we've got
            print(ncfile)
            ncfile.close()
            logger.info('Dataset is closed!')

            for y in range(self.var.shape[0]):
                logger.debug(f'{y = }')
                try: ncfile.close()  # just to be safe, make sure dataset is not already open.
                except: pass
                ncfile = Dataset('ERA5/'+FitKind+'/'+self.prefix+self.name+'_detrended_fields_'+str(self.start_year+y)+'.nc',mode='w',format='NETCDF4_CLASSIC') 
                logger.debug(ncfile)
                lat_dim = ncfile.createDimension('lat', len(lat))     # latitude axis
                lon_dim = ncfile.createDimension('lon', len(lon))    # longitude axis
                time_dim = ncfile.createDimension('time', None) # unlimited axis (can be appended to).
                lat_det = ncfile.createVariable('lat', self.np_precision, ('lat',))
                lat_det.units = 'degrees_north'
                lat_det.long_name = 'latitude'
                lon_det = ncfile.createVariable('lon', self.np_precision, ('lon',))
                lon_det.units = 'degrees_east'
                lon_det.long_name = 'longitude'
                time_det = ncfile.createVariable('time', self.np_precision, ('time',))
                time_det.units = 'days since 1900-01-01'
                time_det.long_name = 'time'
                # Define a 3D variable to hold the data
                NC_det = ncfile.createVariable(self.name+'_det',self.np_precision,('time','lat','lon')) # note: unlimited dimension is leftmost
                NC_det.standard_name = 'detrended '+self.label # this is a CF standard name

                lat_det[:] = lat
                lon_det[:] = lon
                time_det = self.time[y]
                NC_det[:,:,:] = self.detrended[y]

                # first print the Dataset object to see what we've got
                # print(ncfile)
                ncfile.close()
                logger.debug('Dataset is closed!')
        #### In the end compute the means #####
        self.detr_mean = np.mean(self.detrended,0)    # Evaluate detrended grid-point climatological mean
        self.detr_std = np.std(self.detrended,0)    # Evaluate detrended grid-point climatological std  
        
    def Set_detr_area_integral(self, input_area, input_mask):   # Evaluate area integral over the detrended field
        series = np.zeros((self.var.shape[0],self.var.shape[1]), dtype=self.np_precision)
        anomaly_series = np.zeros((self.var.shape[0],self.var.shape[1]), dtype=self.np_precision)
        for y in range(self.var.shape[0]):
            for i in range(self.var.shape[1]):
                self.detr_mask[y,i] = np.sum(np.sum(create_mask(self.Model,input_area,self.detrended[y,i,:,:])*input_mask))
        series = self.detr_mask.copy()
        self.ano_mask = self.detr_mask - np.mean(self.detr_mask,0)    # Evaluate detrended grid-point climatological mean 
        anomaly_series = self.ano_mask.copy()
        return series, anomaly_series

@ut.execution_time
def monotonize_years(da:xr.DataArray):
    '''
    Transforms the time coordinate such that the years are consecutive and increasing monotonically and starting from 0.

    Parameters
    ----------
    da : xr.DataArray
        array with the data. Must have a 'time' dimension

    Returns
    -------
    da : xr.DataArray
        data with monotonized years
    yrs : int
        number of years in the dataset
    '''
    @np.vectorize
    def change(a, y):
        return a.replace(year = y)

    old_y = np.array(da.time.dt.year)
    y = None
    new_y = 0

    old_y_list = [old_y[0]]
    new_ys = np.zeros_like(old_y, dtype=int)
    for i, _y in enumerate(old_y):
        if y is None:
            y = _y
        if _y != y: # new year detected
            y = _y
            old_y_list.append(y)
            new_y += 1
        new_ys[i] = new_y

    # check if the years are already sorted
    old_y_list = np.array(old_y_list)
    if not (old_y_list[1:] - old_y_list[:-1] > 0).all(): # sort years
        new_ys = change(da.time, new_ys)
        da = da.assign_coords({'time': new_ys})

    return da, new_y + 1

@ut.execution_time
def monotonize_longitude(da:xr.DataArray):
    '''
    Makes the leongitude of an array monotonic. This is useful when working with rolled data.

    For example if the original longitude is
        [120, 180, 240, 300, 0, 60]
    The monotonized one will be
        [-240, -180, -120, -60, 0, 60]

    Parameters
    ----------
    da : xr.DataArray
        original data

    Returns
    -------
    xr.DataArray
        data with monotonized longitude
    '''
    slon = np.sign(da.lon.data[1:] - da.lon.data[:-1])
    if len(set(list(slon))) == 1: # no sign change
        return da
    
    new_zero = 360 - da.lon.data[0]
    return da.assign_coords({'lon': (da.lon + new_zero) % 360 - new_zero})



#### masked average of a field ####

def masked_average(xa:xr.DataArray,
                   dim=None,
                   weights:xr.DataArray=None,
                   mask:xr.DataArray=None) -> xr.DataArray:
    '''
    Computes the average of `xa` over given dimensions `dim`, weighting with `weights` and masking with `mask`

    Parameters
    ----------
    xa : xr.DataArray
        data
    dim : str or list of str, optional
        dimensions over which to perform the average, by default None
    weights : xr.DataArray, optional
        weights for the average, for example the cell, by default None
    mask : xr.DataArray, optional
        True over the data to keep, False over the data to ignore, by default None

    Returns
    -------
    xr.DataArray
        masked and averaged array
    '''
    if weights is not None:
        _weights = weights.copy()
        if mask is not None:
            _weights = _weights.where(mask, 0)
    elif mask is not None:
        _weights = xr.where(mask, 1, 0)
    else: # mask = weights = None
        return xa.mean(dim=dim)

    _weights /= _weights.sum(dim=dim) # normalize weights
    _xa = xa*_weights
    return _xa.sum(dim=dim)

def weight_average(*a,**kwargs):
    '''
        Perform weighted average
    '''
    if not isinstance(kwargs['axis'], int): # if it is not an integer, then it is probably a tuple
        if isinstance(kwargs['axis'], tuple):  
            kwargs['axis'] = kwargs['axis'][0] # Otherwise what is being fed into this function could be something of the sort (dim,), which is in conflict with weights
        else:
            raise ValueError(f"axis must be an integer or a tuple, but it is {kwargs['axis'] = }")
    return np.average(*a,**kwargs)

def rolling_reduce_weighted(my_a:xr.DataArray, T, weights=None):
    '''
        Perform weighted average
    '''
    #print(f"{my_a = }")
    rolling = my_a.rolling(time=T, center=False)
    #print(f"{rolling = }")
    rolling.construct("window_dim")
    #print(f"{rolling = }")
    return rolling.reduce(weight_average, weights=weights).dropna('time')*np.sum(weights)/len(weights)

def running_mean(xa:xr.DataArray, T, mode='forward', separate_years=True, weights=None):
    '''
    Computes the running mean over the time axis of an array of data

    Parameters
    ----------
    xa : xr.DataArray
        data, must have a 'time' dimension
    T : int
        width of the window for averaging in time units of the data. If the data is daily, than `T` will be in days.
    mode : 'forward', 'backward' or 'center', optional
        To which point to assign the value of the mean:
            'forward': the value of the mean is assigned to the first point of the window, i.e. we are computing the forward T day average
            'backward': the value of the mean is assigned to the last point of the window
            'center': the value of the mean is assigned to the center point of the window
        By default 'forward'
    separate_years : bool, optional
        Whether to treat each year independently, by default True

    Returns
    -------
    xr.DataArray
        time averaged data

    Raises
    ------
    ValueError
        If invalid `mode`
    '''
    # define the averaging function
    if mode == 'backward':
        if weights is not None:
            raise ValueError(f"Currently weighted running mean only suppported with mode = 'forward'")
        else:
            t_avg = lambda a: a.rolling(time=T, center=False).mean().dropna('time')
        #                                                        ^ this removes nan values
    elif mode == 'forward':
        if weights is None:
            t_avg = lambda a: a[::-1].rolling(time=T, center=False).mean().dropna('time')[::-1]
        # we work on the reversed array so we have the forward T day rolling mean, since xarray by default computes the backward rolling mean
    elif mode == 'center':
        if weights is not None:
            raise ValueError(f"Currently weighted running mean only suppported with mode = 'forward'")
        else:
            t_avg = lambda a: a.rolling(time=T, center=True).mean().dropna('time')
    else:
        raise ValueError(f"Unrecognized {mode = }. Possible options are 'forward', 'backward' or 'center'")

    if separate_years:
        if weights is not None:
            return xa.groupby('time.year').apply(rolling_reduce_weighted, T=T, weights=weights)
        else:
            return xa.groupby('time.year').apply(t_avg) # apply to each year individually
    if weights is not None:
        return rolling_reduce_weighted(xa, T=T, weights=weights)
    else:
        return t_avg(xa)

def is_over_threshold(a:np.ndarray, threshold=None, percent=None):
    '''
    Computes whether data are above a given threshold or percentile

    Parameters
    ----------
    a : np.ndarray
        data
    threshold : float, optional
        threshold, by default None
    percent : float in [0, 100], optional
        Percentile used to compute the threshold, ignored if `threshold` is provided, by default None.
        For example `percent` = 5 means that the threshold will be the ones that leaves 5 percent of the data above it

    Returns
    -------
    l : np.ndarray
        array of bools of the same shape as `a` indicating which datapoints are >= `threshold`
    threshold : float

    Raises
    ------
    ValueError
        If both `threshold` and `percent` are None
    '''
    if threshold is None:
        if percent:
            a_flat = a.flatten()
            threshold = np.sort(a_flat)[np.ceil(a_flat.shape[0]*(1-percent/100)).astype('int')]
        else:
            raise ValueError('Please provide threshold or percent')
    return a >= threshold, threshold

def pretty_set_of_int(s:set) -> str:
    '''
    Takes a set of int as input and summarizes it in a string.
    For example {1,2,3,5} -> '1-3, 5'
    '''
    yr = np.sort(list(s))
    diffs = np.insert(yr[1:] - yr[:-1], 0, 0)
    intervals = {}
    prev_start = None
    for i,d in enumerate(diffs):
        if d != 1:
            if prev_start is not None:
                intervals[prev_start] = yr[i-1]
            prev_start = yr[i]
    intervals[prev_start] = yr[-1]
    return ', '.join([f'{start}' + (f'-{end}' if end != start else '') for start,end in intervals.items()])
    
class Plasim_Field:
    def __init__(self, name, filename, label, Model, years=None, mylocal='/local/gmiloshe/PLASIM/', **kwargs):
        self.name = name    # Name inside the .nc file
        self.filename = filename # path to the .nc file starting from `mylocal`
        self.label = label  # Label to be displayed on the graph
        self.Model = Model
        self.years = years
        self.mylocal = mylocal

        self.mask_area = None
        self.mask = None
        self.A = None # This is a placeholder variable that can be used as a running mean save. It might be unnecessary given that there is an existing attribute:  _area_integral


        logger.info(f'Opening field {self.name}')
        
        try:
            self.datas = xr.open_dataset(ut.first_valid_path(self.mylocal,self.filename)) #.fillna(0) # The issue is that Francesco put a land mask on TAS.nc which has nan values on the sea.
                                # For machine learning purposes nan could be a problem
            self.field = standardize_dim_names(self.datas[name])
        except KeyError:
            logger.error(f'Unable to find key "{name}" among the provided fields {list(self.datas.keys())}')
            raise KeyError
        
        self.field = discard_all_dimensions_but(self.field, dims_to_keep=['time', 'lon', 'lat'])
        
        self.field, yrs = monotonize_years(self.field)
        if yrs != self.years:
            if self.years is not None:
                logger.error(f'The loaded field has {yrs} years, not {years} as provided. Setting self.years = {yrs}')
            self.years = yrs

        self.land_area_weights = get_lsm(self.mylocal,self.Model).sel(lat=self.field.lat, lon=self.field.lon)
        self.area_weights = get_cell_area(self.mylocal, self.Model).sel(lat=self.field.lat, lon=self.field.lon)
        
        self.land_area_weights.data *= self.area_weights.data
        self.area_weights.data /= np.sum(self.area_weights.data)
        self.land_area_weights.data /= np.sum(self.land_area_weights.data)

        self._area_integral = None
        self._time_average = None

    @property
    def year_range(self):
        return pretty_set_of_int(set(self.field.time.dt.year.data))

    @ut.execution_time
    def select_years(self, year_list=None):
        '''
        Select a subset of years

        Parameters
        ----------
        year_list : array-like, optional
            list of the years to keep, by default None
        '''
        # check if the given year list is within the range of the data
        if year_list is not None:
            invalid_years = set(year_list) - set(self.field.time.dt.year.data)
            if invalid_years:
                raise IndexError(f'Data year range is {self.year_range} which does not include {pretty_set_of_int(invalid_years)}')
            self.field = self.field.sel(time=self.field.time.dt.year.isin(year_list))
            self.years = len(year_list)
    
    @ut.execution_time
    def sort_lat(self):
        '''
        Sorts the latitudes so that they always increase from the North Pole to the South Pole
        This is done so that the latitude order be `Model` indepdendent
        '''
        _latitudes = self.field.lat
        _latitudes_sorted = _latitudes.sortby(_latitudes, ascending=False)
        if (_latitudes == _latitudes_sorted).all():
            return
        self.field = self.field.sel(lat=_latitudes_sorted)
        self.area_weights = self.area_weights.sel(lat=self.field.lat)
        self.land_area_weights = self.land_area_weights.sel(lat=self.field.lat)
        if self.mask is not None:
            self.mask = self.mask.sel(lat=self.field.lat)

    @ut.execution_time
    @ut.indent_logger(logger)
    def select_lonlat(self, lat_start=None, lat_end=None, lon_start=None, lon_end=None, fillna=None):
        '''
        Select a region in space.
        If `lon_start` >= `lon_end` the selection will start from `lon_start`, go over the end of the array and then continue from the beginning up to `lon_end`.
        Providing `lon_start` = `lon_end` will result in the longitude being rolled by `lon_start` steps

        Parameters
        ----------
        lat_start : int, optional
            start index for latitude, by default None
        lat_end : int, optional
            end index for latitude, by default None
        lon_start : int, optional
            start index for longitude, by default None
        lon_end : int, optional
            end index for longitude, by default None
        fillna : float, optional
            value to fill the missing values with, by default None
        '''
        if lat_start or lat_end:
            self.field = self.field.isel(lat=slice(lat_start, lat_end))
            self.area_weights = self.area_weights.sel(lat=self.field.lat)
            self.land_area_weights = self.land_area_weights.sel(lat=self.field.lat)
            if self.mask is not None:
                self.mask = self.mask.sel(lat=self.field.lat)
                
        if lon_start or lon_end:
            concatenation = False
            if lon_start and lon_end:
                lon_start = lon_start % len(self.field.lon)
                lon_end = lon_end % len(self.field.lon)
                if lon_start >= lon_end:
                    concatenation = True

            if concatenation:
                self.field = xr.concat([self.field.isel(lon=slice(lon_start,None)),self.field.isel(lon=slice(None,lon_end))], dim='lon') # this loads the field in memory
                # self.field.assign_coords({'lon': (self.field.lon + 180) % 360 - 180})
            else:
                self.field = self.field.isel(lon=slice(lon_start, lon_end))

            self.area_weights = self.area_weights.sel(lon=self.field.lon)
            self.land_area_weights = self.land_area_weights.sel(lon=self.field.lon)
            if self.mask is not None:
                self.mask = self.mask.sel(lon=self.field.lon)

        if fillna is not None:
            logger.info(f'Filling missing values with {fillna}')
            self.field = self.field.fillna(fillna)
        else:
            logger.info('No filling of missing values')

    @property
    def var(self):
        '''Gives access to the data of the field. For compatibility with the old version'''
        return self.to_numpy(self.field)

    def to_numpy(self, da:xr.DataArray):
        '''Returns a numpy version of `da`, reshaping the time axis in years (the number is given by self.years) and days of year'''
        data_shape = da.shape
        if data_shape[0] % self.years:
            logger.warning(f'Cannot reshape time axis of field {da.name}')
            return da.data
        return da.data.reshape((self.years, data_shape[0]//self.years, *data_shape[1:]))

    def set_mask(self, area):
        '''
        Sets a mask for the object. The mask is adapted to past and future coordinate transformations of the data.

        Parameters
        ----------
        area : str
            name of the mask area, must be one of the ones allowed by the function `create_mask`
        '''
        self.mask_area = area
        self._area_integral = None
        self.mask = get_lsm(self.mylocal,self.Model)
        try:
            self.mask = create_mask_xarray(self.Model,area, self.mask)
        except:
            logger.warning('Failed to create mask with xarray features: using old version with numpy')
            self.mask.data = create_mask(self.Model,area,self.mask.data, axes='last 2', return_full_mask=True)

        self.mask = self.mask.sel(lat=self.field.lat, lon=self.field.lon)
        

    def filter(self, keep_inside_mask=True):
        '''If `keep_inside_mask` sets to zero all values of self.field outside the mask. Otherwise the ones inside.'''
        if self.mask is None:
            raise ValueError('Mask not set: cannot filter. Please use `self.set_mask`')
        if self.field.shape[1:] != self.mask.shape:
            raise ValueError(f'Mismatched shapes: {self.field.shape[1:] = }, {self.mask.shape = }')

        if keep_inside_mask:
            self.field.data *= self.mask.data
        else:
            self.field.data *= np.logical_not(self.mask.data)
    
    @ut.execution_time
    def compute_area_integral(self, weights='land_area'):
        '''
        Computes the area integral over the region set by the mask and stores it in self._are_integral

        Parameters
        ----------
        weights : xr.DataArray or 'area' or 'land_area', optional
            Weights to use for the average, by default 'land_area'
            If provided as a xr.DataArray it must have the same grid as self.field, other (recommended) options are
                'land_area': weights are the grid cell area over land masses
                'area': weights are the grid cell area (over land and sea)

        Returns
        -------
        self._area_integral : xr.DataArray
            area intgral
        '''
        if isinstance(weights, str):
            if weights == 'area':
                weights = self.area_weights
            elif weights == 'land_area':
                weights = self.land_area_weights
            else:
                raise ValueError(f'Unrecognized string option {weights = }')
        elif not isinstance(weights, xr.DataArray):
            raise TypeError(f'weights must be either string or xr.DataArray, not {type(weights)}')

        self._area_integral = masked_average(self.field, dim=['lat', 'lon'], weights=weights, mask=self.mask)
        return self._area_integral

    @property
    def area_integral(self):
        '''Access to the area integral. If it has not been computed yet it is computed with default weights'''
        if self._area_integral is None:
            self.compute_area_integral()
        return self._area_integral

    @ut.execution_time
    @ut.indent_logger(logger)
    def compute_time_average(self, day_start, day_end, T, weights=None):
        '''
        Computes the forward running mean of the self._area_intgral attribute

        Parameters
        ----------
        day_start : int
            first day of the year to consider (1 means 1st of January)
        day_end : int
            (last day of the year to consider) + 1, as usual with slicing
        T : int
            width in time units of the time average

        Returns
        -------
        np.ndarray
            time average
        '''
        cut_area_integral = self.area_integral.sel(time=self.area_integral.time.dt.dayofyear.isin(np.arange(day_start, day_end)))
        self._time_average = running_mean(cut_area_integral, T, mode='forward',weights=weights) # cache the time average
        return self._time_average

        

    
class Plasim_Field_Old:
    def __init__(self, name, filename, label, Model, lat_start=0, lat_end=64, lon_start=0, lon_end=128,
                 myprecision='double', mysampling='', years=1000):
        self.name = name    # Name inside the .nc file
        self.filename = filename # Name of the .nc file 
        self.label = label  # Label to be displayed on the graph
        self.var = []    
        self.lat_start = lat_start
        self.lat_end = lat_end
        self.lon_start = lon_start
        self.lon_end = lon_end
        self.Model = Model
        self.years = years # years must be the correct number of years in the dataset # GM: There is probably a way to just read the years from *.nc
        self.sampling = mysampling # This string is used to distinguish daily vs 3 hr sampling, which is going to be important when we compute area integrals. The idea is that if we have already computed daily we must change the name for the new 3 hrs sampling file, otherwise the daily file will be loaded (see the routine Set_area_integral)
        if myprecision == 'double':
            self.np_precision = np.float64
            self.np_precision_complex = np.complex128
        else: # we can save space since for learning we don't need as much precision
            self.np_precision = np.float32
            self.np_precision_complex = np.complex64
        
    @ut.execution_time
    @ut.indent_logger(logger)
    def load_field(self, folder, year_list=None):
        '''
        Load the file from the database stored in `folder`
        
        `year_list` allows to load only a subset of data. If not provided all years are loaded
        '''
        logger.info(f'Loading field {self.name}')
        if self.sampling == '3hrs':
            self.var = np.zeros((self.years,1200,self.lat_end-self.lat_start,self.lon_end-self.lon_start), dtype=self.np_precision)
            for b in range(1, self.years//100 + 1):
                logger.debug("b = ",b)
                for y in range(1,101):
                    for m in range(5,10):
                        temp_time, temp_var = self.load_month(folder,b,y,m)
                        self.var[100*(b-1)+y-1,240*(m-5):240*(m-4)] = temp_var

            self.varmean = np.mean(self.var,0)
            self.var -= self.varmean  # to get climatological anomalies out of original fields
            
        else:
            dataset = Dataset(folder+self.filename+'.nc')
            if year_list is None: # load all years
                self.time = np.asarray(dataset.variables['time']).reshape(self.years,-1) # CHANGE TO XARRAY
            else:
                units_per_year = dataset.variables['time'].shape[0]//self.years
                if dataset.variables['time'].shape[0] != self.years*units_per_year:
                    raise ValueError(f"{self.filename} doesn't have {self.years} years") 
                self.time = np.concatenate([dataset.variables['time'][y*units_per_year:(y+1)*units_per_year,...] for y in year_list], axis=0).reshape(len(year_list),-1)
            logger.info('Loaded time array')
            if (self.name == 'zg') or (self.name == 'ua') or (self.name == 'va'): # we need to take out dimension that is useless (created by extracting a level)
                if year_list is None:
                    self.var = np.asarray(dataset.variables[self.name][:,0,self.lat_start:self.lat_end,self.lon_start:self.lon_end],  dtype=self.np_precision)
                else:
                    self.var = [np.asarray(dataset.variables[self.name][y*units_per_year:(y+1)*units_per_year,0,self.lat_start:self.lat_end,self.lon_start:self.lon_end],  dtype=self.np_precision) for y in year_list]
                    self.var = np.concatenate(self.var, axis=0)
            else: 
                if year_list is None:
                    self.var = np.asarray(dataset.variables[self.name][:,self.lat_start:self.lat_end,self.lon_start:self.lon_end],  dtype=self.np_precision)
                else:
                    self.var = [np.asarray(dataset.variables[self.name][y*units_per_year:(y+1)*units_per_year,self.lat_start:self.lat_end,self.lon_start:self.lon_end],  dtype=self.np_precision) for y in year_list]
                    self.var = np.concatenate(self.var, axis=0)
            logger.info(f"input {self.var.shape = }")
            if year_list is None:
                self.var = self.var.reshape(self.years, self.var.shape[0]//self.years, *self.var.shape[1:])
            else:
                self.var = self.var.reshape(len(year_list), units_per_year, *self.var.shape[1:])

            self.lon = dataset.variables["lon"][self.lon_start:self.lon_end]
            self.lat = dataset.variables["lat"][self.lat_start:self.lat_end]
            self.LON, self.LAT = np.meshgrid(self.lon, self.lat)
            logger.info(f"output {self.var.shape = }")
            logger.debug(f"{self.time.shape = }")
            logger.debug(f'{np.min(np.diff(self.time))} < np.diff(self.time) < {np.max(np.diff(self.time))}')
            dataset.close()
            
            
        
    def load_month(self, folder,century,year,month):   # Load individual months CAREFUL: parameters are defined from 1 to 12 not from 0 to 11!!!
        nb_zeros_m = 2-len(str(month))  #we need to adjust the name of the file we are addressing
        nb_zeros_c = 4-len(str(century))
        auto_name = "CONTROL_BATCH"+nb_zeros_c*"0"+str(century)+"_"+self.filename+".0"+str(month)
        path = folder+auto_name+"/"
        nb_zeros_y = 4 - len(str(year))
        filename = auto_name+"."+nb_zeros_y*"0"+str(year)+".nc"
        dataset = Dataset(path+filename)
        local_time = np.asarray(dataset.variables['time'][:])
        if self.name == 'zg':
            local_var = np.asarray(dataset.variables[self.name][:,0,self.lat_start:self.lat_end,self.lon_start:self.lon_end],  dtype=self.np_precision)
        else:
            local_var = np.asarray(dataset.variables[self.name][:,self.lat_start:self.lat_end,self.lon_start:self.lon_end],  dtype=self.np_precision)
        
        self.lon = dataset.variables["lon"][self.lon_start:self.lon_end]
        self.lat = dataset.variables["lat"][self.lat_start:self.lat_end]
        self.LON, self.LAT = np.meshgrid(self.lon, self.lat)
        
        dataset.close()
        return local_time, local_var
        
    def Greenwich(self, year, day):  # Take an array and copy the Greenwhich meridian to make sure we avoid a segment hole in basemap
        '''
        Returns `self.var` with the Greenwich meridian counted twice (start and end of the array): useful for plotting
        
        Parameters:
        year: int or 'all' (this last one only if `day` == 'all')
        day: int or 'all'
        '''
        # old
        # return np.append(self.var[year,day,:,:],np.transpose([self.var[year,day,:,0]]),axis = 1)
        if day == 'all':
            if year == 'all':
                return Greenwich(self.var)
            return Greenwich(self.var[year])
        if year == 'all':
            raise NotImplementedError('if you specify a day, you must also specify a year')
        return Greenwich(self.var[year,day])
    
    @ut.execution_time
    def Set_area_integral(self, input_area, input_mask, containing_folder='Postproc', delta=1, force_computation=False):
        '''
        Evaluate area integral and (possibly if delta is not 1) coarse grain it in time
        
        if `force_computation` == True, the integrals are computed in any case.
        Otherwise, if the arrays with the integrals are found in `containing_folder` they are loaded rather than computed
        If `containing_folder` is None or False the area integrals are not saved
        '''
        if containing_folder:
            if delta == 1:
                filename_abs =  f'{containing_folder}/Int_Abs_{self.sampling}_{self.Model}_{input_area}_{self.filename}.npy'
                filename_ano_abs =  f'{containing_folder}/Int_Ano_Abs_{self.sampling}_{self.Model}_{input_area}_{self.filename}.npy'
            else:
                filename_abs =  f'{containing_folder}/Int_Abs_{self.sampling}_{self.Model}_{input_area}_{self.filename}_{delta}.npy'
                filename_ano_abs =  f'{containing_folder}/Int_Ano_Abs_{self.sampling}_{self.Model}_{input_area}_{self.filename}_{delta}.npy'
            
        if (not force_computation) and containing_folder and os.path.exists(filename_abs): # load integrals
            self.abs_mask = np.load(filename_abs)
            self.ano_abs_mask = np.load(filename_ano_abs)
            logger.info(f'file {filename_abs} loaded')
            logger.info(f'file {filename_ano_abs} loaded')
        else: # compute integrals            
            self.abs_mask = np.tensordot(create_mask(self.Model,input_area,self.var, axes='last 2'), input_mask) # potential BUG since self.var is cut in longitude and latitude when loaded
            
            #print("self.ano_abs_mask = self.abs_mask - np.mean(self.abs_mask,0)")
            self.ano_abs_mask = self.abs_mask - np.mean(self.abs_mask,0)    # Evaluate grid-point climatological mean 
            if delta > 1:
                for obj in [self.ano_abs_mask, self.abs_mask]: # do it for both objects
                    A = np.zeros((obj.shape[0], obj.shape[1] - delta + 1), dtype=self.np_precision)   # When we use convolve (running mean) there is an extra point that we can generate by displacing the window hence delta - 1 instead of delta
                    convseq = np.ones(delta)/delta
                    for y in range(obj.shape[0]):
                        A[y,:]=np.convolve(obj[y,:],  convseq, mode='valid')
                    obj = A
            # if not keep the definitions of the objects

            # create containing folder if it doesn't exist
            if containing_folder:
                if not os.path.exists(containing_folder):
                    containing_folder = Path(containing_folder).resolve()
                    containing_folder.mkdir(parents=True,exist_ok=True)

                np.save(filename_abs,self.abs_mask)
                np.save(filename_ano_abs,self.ano_abs_mask)
                logger.info(f'saved file {filename_abs}')
                logger.info(f'saved file {filename_ano_abs}')
        anomaly_series = self.ano_abs_mask.copy()
        series = self.abs_mask.copy()
        
        return series, anomaly_series
    
    def PreMixing(self, new_mixing, containing_folder='Postproc', num_years=None, select_group=0):
        ''''
        Randomly permute all years (useful for Machine Learning input), mixes the batches but not the days of a year! num_years - how many years are taken for the analysis
        
        WARNING: modifies the object attributes, e.g. self.var
        '''
        if num_years is None: 
            num_years = self.years
        #print(type(containing_folder),type(self.sampling), type(self.Model))
        logger.info(f"{containing_folder = }, {self.sampling = }, {self.Model = }")
        filename = f'{containing_folder}/PreMixing_{self.sampling}_{self.Model}.npy'
        if ((new_mixing) or (not os.path.exists(filename))): # if we order new mixing or the mixing file doesn't exist
            mixing = np.random.permutation(self.var.shape[0])
            np.save(filename, mixing)
            logger.info(f'saved file {filename}')
        else:
            mixing = np.load(filename)
            logger.info(f'file {filename} loaded')
        
        logger.info(f"{mixing.shape = }")
        mixing = mixing[num_years*select_group:num_years*(select_group+1)] # This will select the right number of years
        logger.info(f"Selected group {select_group}: {mixing.shape = }")
        self.var = self.var[mixing,...]  # This will apply permutation on all years
        logger.info(f'mixed {self.var.shape = }')
        if hasattr(self, 'abs_mask'):
            self.abs_mask = self.abs_mask[mixing,...]
            logger.info(f'mixed {self.abs_mask.shape = }')
        if hasattr(self, 'ano_abs_mask'):
            self.ano_abs_mask = self.ano_abs_mask[mixing,...]
            logger.info(f'mixed {self.ano_abs_mask.shape = }')
        if hasattr(self, 'abs_area_int'):
            self.abs_area_int = self.abs_area_int[mixing,...]
            logger.info(f'mixed {self.abs_area_int.shape = }')
        if hasattr(self, 'ano_area_int'):
            self.ano_area_int = self.ano_area_int[mixing,...]
            logger.info(f'mixed {self.ano_area_int.shape = }')
            
            
        self.new_mixing = new_mixing
        #self.time = self.time[mixing,...]   <- This we can't use because I don't load time in 3hrs sampling case
        
        return filename
    
    
    def EqualMixing(self, A, threshold, new_mixing, containing_folder='Postproc', num_years=1000, select_group=0, delta=1, threshold_end=''): 
        '''
        Permute all years (useful for Machine Learning input), mix until each batch has the same number of heatwave days!
        '''
        
        if str(threshold) != '2.953485': # use new labeling # GEORGE: there is indeed a way to remove this awkward statement. This is old threshold for Plasim 1000 years dataset that dates back to the time when I didn't specify threshold in the mixing file. This threshold is obtained if we take 5 percent heatwaves over France. The idea was to default in this case to the old equal mixing and avoid creating a new permutation. What can be done instead is to simply copy the old file and give it the appropriate name given this new system where we have to add a threshold in the filename
            filenamepostfix1 = '_'+str(threshold)
        else:
            filenamepostfix1 = ''
        if num_years != 1000: # use new labeling
            filenamepostfix2 = '_'+str(num_years)
        else:
            filenamepostfix2 = ''
        if select_group == 0:
            filenamepostfix3 = ''
        else:
            filenamepostfix3 = '_'+str(select_group)
        if delta == 1:
            filenamepostfix4 = ''
        else:
            filenamepostfix4 = '_'+str(delta)
        if threshold_end == '': # This is reserved in case we want to define extremes between two thresholds
            filenamepostfix5 = ''
        else:
            filenamepostfix5 = '_'+str(threshold_end)
        filename = f'{containing_folder}/EqualMixing_{self.sampling}_{self.Model}{filenamepostfix1}{filenamepostfix2}{filenamepostfix3}{filenamepostfix4}{filenamepostfix5}.npy'
        
        if ((new_mixing) or (not os.path.exists(filename))): # if we order new mixing or the mixing file doesn't exist
            if threshold_end == '': # If we don't provide the end we imply that it is max of A
                mixed_event_per_year = np.sum((A>=threshold),1)
            else:   # If we provide the threshold_end we expect it to be the upper cap on the heatwaves
                mixed_event_per_year = np.sum((A>=threshold)&(A<threshold_end),1)
            mixing = np.arange(A.shape[0])
            entropy_per_iteration, number_per_century, norm_per_century = ComputeEntropy(mixed_event_per_year,mixing)

            #number_per_century=np.sum(mixed_event_per_year.reshape((10,-1)),1)#/ (A.shape[1]*A.shape[0]//10)
            #norm_per_century=number_per_century/np.sum(number_per_century)
            #entropy_per_iteration = -np.sum(norm_per_century*np.log(norm_per_century))
            logger.info(f"{number_per_century = }")
            logger.info(f"{entropy_per_iteration = }, normalization = {np.sum(number_per_century)}")

            for myiter in range(10000000):
                #print("========")
                #print(myiter)
                randrange1=randrange(A.shape[0])
                for testing in range(10):
                    randrange2=randrange(A.shape[0])
                    if randrange2 != randrange1:
                        break
                #print("permute from ",randrange1," with ", mixed_event_per_year[randrange1], " events to ", randrange2, " with", mixed_event_per_year[randrange2])

                #mixing[randrange1] = randrange2
                #mixing[randrange2] = randrange1
                #mixing =  PermuteFullrange(mixing, randrange1, randrange2)
                number_per_century_old = number_per_century
                entropy_per_iteration_prime, number_per_century, norm_per_century = ComputeEntropy(mixed_event_per_year,PermuteFullrange(mixing, randrange1, randrange2))
                if entropy_per_iteration_prime > entropy_per_iteration:
                    oldmixing = mixing
                    mixing = PermuteFullrange(mixing, randrange1, randrange2) # we actually permute
                    entropy_per_iteration_prime, number_per_century, norm_per_century = ComputeEntropy(mixed_event_per_year,mixing)
                    mixingdublicatenumber = (len([item for item, count in collections.Counter(mixing).items() if count > 1]))
                    if mixingdublicatenumber == 1:
                        logger.debug(randrange1,randrange2)
                        logger.debug(oldmixing)
                        logger.debug(mixing)
                        logger.debug(([item for item, count in collections.Counter(mixing).items() if count > 1]))
                    logger.debug(f"{myiter = }, entropy = {entropy_per_iteration_prime} > {entropy_per_iteration} , #/century = {np.sum(number_per_century)} duplicate# = {mixingdublicatenumber}")
                    entropy_per_iteration = entropy_per_iteration_prime
                #else:
                    #print(entropy_per_iteration_prime," <= ", entropy_per_iteration, " => Keep old!")
                    #mixing[randrange1] = randrange1
                    #mixing[randrange2] = randrange2
                #    mixing =  PermuteFullrange(mixing, randrange1, randrange2)
                #entropy_per_iteration_prime, number_per_century, norm_per_century = ComputeEntropy(mixed_event_per_year,mixing)
                #print("new number_per_century = ", number_per_century)
                #print("new entropy_per_iteration = ", entropy_per_iteration, " ,new sum = ", np.sum(number_per_century))

            number_per_century=np.sum(mixed_event_per_year[mixing].reshape((10,-1)),1)#/ (A.shape[1]*A.shape[0]//10)
            norm_per_century=number_per_century/np.sum(number_per_century)
            entropy_per_iteration_prime=-np.sum(norm_per_century*np.log(norm_per_century))
            logger.info(f"final {number_per_century = }")
            logger.info(f"final {entropy_per_iteration = }, final sum = {np.sum(number_per_century)}")
            np.save(filename, mixing)
            logger.info(f'saved file {filename}')
        else:
            mixing = np.load(filename)
            logger.info(f'file {filename} loaded')
            
        self.var = self.var[mixing,...]  # This will apply permutation on all years
        logger.info(f'mixed {self.var.shape = }')
        if hasattr(self, 'abs_mask'):
            self.abs_mask = self.abs_mask[mixing,...]
            logger.info(f'mixed {self.abs_mask.shape = }')
        if hasattr(self, 'ano_abs_mask'):
            self.ano_abs_mask = self.ano_abs_mask[mixing,...]
            logger.info(f'mixed {self.ano_abs_mask.shape = }')
        if hasattr(self, 'abs_area_int'):
            self.abs_area_int = self.abs_area_int[mixing,...]
            logger.info(f'mixed {self.abs_area_int.shape = }')
        if hasattr(self, 'ano_area_int'):
            self.ano_area_int = self.ano_area_int[mixing,...]
            logger.info(f'mixed {self.ano_area_int.shape = }')
            
        self.new_equalmixing = new_mixing
        #self.time = self.time[mixing,:]   <- This we can't use because I don't load time in 3hrs sampling case
        
        return filename
    
    def ReshapeInto1Dseries_old(self, area, mask, time_start, time_end, T, tau): # Reshape years and days into a 1D series useful for learning. Compatibility with the old version
        filename = 'Postproc/Int_Reshape_'+self.sampling+'_'+self.Model+'_'+area+'_'+self.filename+'_'+str(time_start)+'_'+str(time_end)+'_'+str(T)+'_'+str(tau)+'.npy'
        if os.path.exists(filename):
            #print("series = np.load(filename)")
            series = np.load(filename)
            logger.info(f'file {filename} loaded')
        else:
            self.abs_area_int, self.ano_area_int = self.Set_area_integral(area, mask)
            series = self.abs_area_int[:,(time_start+tau):(time_end+tau - T+1)].reshape((self.abs_area_int[:,(time_start+tau):(time_end+tau - T+1)].shape[0]*self.abs_area_int[:,(time_start+tau):(time_end+tau - T+1)].shape[1]))
            np.save(filename,series)
            logger.info(f'saved file {filename}')
        return series
    
    def ReshapeInto1Dseries(self, area, mask, time_start, time_end, T, tau): # Reshape years and days into a 1D series useful for learning. In the new version we don't recalculate or reload existing time series because they could be mixed in advance (the years may be in the mixed order)!
        if hasattr(self, 'abs_area_int'):
            series = self.abs_area_int[:,(time_start+tau):(time_end+tau - T+1)].reshape((self.abs_area_int[:,(time_start+tau):(time_end+tau - T+1)].shape[0]*self.abs_area_int[:,(time_start+tau):(time_end+tau - T+1)].shape[1]))
        else:
            logger.warning('First execute: self.abs_area_int, self.ano_area_int = self.Set_area_integral(area, mask)')
        return series
    
    def ReshapeInto1DseriesCoarse(self, area, mask, time_start, time_end, T, tau, delta): # Reshape years and days into a 1D series useful for learning. Compute several day average (delta is the coarsegraining time)
        if hasattr(self, 'abs_area_int'):
            series = self.abs_area_int[:,(time_start+tau-delta//2):(time_end+tau - T+1+delta//2)]
            logger.info(f"{series.shape = }")
            convseq = np.ones(delta)/delta

            coarseseries = np.zeros((series.shape[0], series.shape[1]+delta), dtype=self.np_precision)
            for y in range(self.var.shape[0]):
                coarseseries[y,:]=np.convolve(series[y,:],  convseq, mode='valid')
            logger.info(f"{coarseseries.shape = }")
            coarseseries = coarseseries.reshape((coarseseries.shape[0]*coarseseries.shape[1]))
            logger.info(f"{coarseseries.shape = }")

        else:
            logger.info('First execute: self.abs_area_int, self.ano_area_int = self.Set_area_integral(area, mask)')
        return series
    
    def ReshapeInto2Dseries(self,time_start,time_end,lat_from,lat_to,lon_from,lon_to,T,tau, dim=1): 
        '''
        Reshapes the time series of the grid into a flat array useful for feeding this to a flat layer of a neural network
        '''
        selfvarshape = self.var[:,(time_start+tau):(time_end+tau - T+1),lat_from:lat_to,lon_from:lon_to].shape
        temp = self.var[:,(time_start+tau):(time_end+tau - T+1),lat_from:lat_to,lon_from:lon_to].reshape((selfvarshape[0]*selfvarshape[1],selfvarshape[2],selfvarshape[3]))
        if dim == 1: # if we intend for the spatial dimension of the output to be 1D
            return temp.reshape((temp.shape[0],temp.shape[1]*temp.shape[2]), order='F') # Fortran order (last index changes first)
        else:        # if we intend for the spatial dimension of the output stay 2D
            return temp
        
    def DownScale(self,time_start,time_end,lat_from,lat_to,lon_from,lon_to,T,tau, dim): # This function coarse grains the the time series of the grid into the required dim
        selfvarshape = self.var[:,(time_start+tau):(time_end+tau - T+1),lat_from:lat_to,lon_from:lon_to].shape
        temp = self.var[:,(time_start+tau):(time_end+tau - T+1),lat_from:lat_to,lon_from:lon_to].reshape((selfvarshape[0]*selfvarshape[1],selfvarshape[2],selfvarshape[3]))
        return resize(temp, (temp.shape[0], dim[0], dim[1]))
    
    def ComputeFFT(self,time_start,time_end,T,tau, myout='complex'): # This function computes the FFT of the .var variable
        selfvarshape = self.var[:,(time_start+tau):(time_end+tau - T+1),:,:].shape
        temp = self.var[:,(time_start+tau):(time_end+tau - T+1),:,:].reshape((selfvarshape[0]*selfvarshape[1],selfvarshape[2],selfvarshape[3]))
        temp = np.array(np.fft.fftn( temp , axes=(1, 2)), dtype = self.np_precision_complex)
        temp = np.fft.fftshift( temp, axes=(1,2))[:,:,self.var.shape[3]//4:3*self.var.shape[3]//4]  # remove a half of the longitude 
        temp = np.concatenate([np.zeros((temp.shape[0],(temp.shape[2]-temp.shape[1])//2,temp.shape[2])),temp,np.zeros((temp.shape[0],(temp.shape[2]-temp.shape[1])//2,temp.shape[2]))],axis=1, dtype = self.np_precision_complex)
        if myout == 'complex': # return as a complex number
            return temp
        else: # return as an array with a last index corresponding to the real and complex part
            temp2 = np.zeros((temp.shape[0],temp.shape[1],temp.shape[2],2), dtype = self.np_precision)
            temp2[:,:,:,0] = np.real(temp)
            temp2[:,:,:,1] = np.imag(temp)
            return temp2
        
    def ComputeFFThalf(self,time_start,time_end,T,tau, myout='complex'): # This function computes the FFT of the .var variable with the half of the resolution
        selfvarshape = self.var[:,(time_start+tau):(time_end+tau - T+1),:,:].shape
        temp = self.var[:,(time_start+tau):(time_end+tau - T+1),:,:].reshape((selfvarshape[0]*selfvarshape[1],selfvarshape[2],selfvarshape[3]))
        temp = np.array(np.fft.fftn( temp , axes=(1, 2)), dtype = self.np_precision_complex)
        temp = np.fft.fftshift( temp, axes=(1,2))[:,:,3*self.var.shape[3]//8:5*self.var.shape[3]//8]  # remove 3 quaters of the longitude 
        temp = np.concatenate([np.zeros((temp.shape[0],(temp.shape[2]-temp.shape[1])//2,temp.shape[2])),temp,np.zeros((temp.shape[0],(temp.shape[2]-temp.shape[1])//2,temp.shape[2]))],axis=1, dtype = self.np_precision_complex)
        if myout == 'complex': # return as a complex number
            return temp
        else: # return as an array with a last index corresponding to the real and complex part
            temp2 = np.zeros((temp.shape[0],temp.shape[1],temp.shape[2],2), dtype = self.np_precision)
            temp2[:,:,:,0] = np.real(temp)
            temp2[:,:,:,1] = np.imag(temp)
            return temp2
    

    def ComputeFFTnoPad(self,time_start,time_end,T,tau, myout='complex', mydim = []): # This function computes the FFT of the .var variable without padding 0s in the latitude direction
        selfvarshape = self.var[:,(time_start+tau):(time_end+tau - T+1),:,:].shape
        temp = self.var[:,(time_start+tau):(time_end+tau - T+1),:,:].reshape((selfvarshape[0]*selfvarshape[1],selfvarshape[2],selfvarshape[3]))
        temp = np.array(np.fft.fftn( temp , axes=(1, 2)), dtype = self.np_precision_complex)
        if mydim == []:
            mydim = [temp.shape[1], temp.shape[1]]
        temp = np.fft.fftshift( temp, axes=(1,2))[:,(temp.shape[1]-mydim[0])//2:(temp.shape[1]+mydim[0])//2,(temp.shape[2]-mydim[1])//2:(temp.shape[2]+mydim[1])//2]  # remove a half of the longitude 
        if myout == 'complex': # return as a complex number
            return temp
        else: # return as an array with a last index corresponding to the real and complex part
            temp2 = np.zeros((temp.shape[0],mydim[0],mydim[1],2), dtype = self.np_precision)
            temp2[:,:,:,0] = np.real(temp)
            temp2[:,:,:,1] = np.imag(temp)
            return temp2

    def ComputeTimeAverage(self,time_start,time_end,T=14,tau=0, percent=5,delta=1, threshold=None): 
        '''
        Computes time average from time series

        `tau` is not used
        if `threshold` is provided, it overrides percent
        '''
        A = np.zeros((self.var.shape[0], time_end - time_start - T + 1), dtype=self.np_precision)   # When we use convolve (running mean) there is an extra point that we can generate by displacing the window hence T-1 instead of T
        if delta==1:
            convseq = np.ones(T)/T
        else: # if coarse graining is applied we define time average differently by skipping unnecessary steps
            convseq = np.zeros(T) # convolution to be used for running mean
            convseq[range(0,T,delta)] = 1/(T//delta)
            convseq = convseq[::-1] # this vector to be used in the convolution has to be inverted for the correct function
            print(f"{convseq = }")
        for y in range(self.var.shape[0]):
            A[y,:]=np.convolve(self.abs_area_int[y,(time_start):(time_end)],  convseq, mode='valid')
        A_reshape = A.reshape((A.shape[0]*A.shape[1]))
        if threshold is None:
            threshold = np.sort(A_reshape)[np.ceil(A_reshape.shape[0]*(1-percent/100)).astype('int')]
        list_extremes = list(A_reshape >= threshold)
        return A, A_reshape, threshold, list_extremes, convseq
        
def ComputeEntropy(mixed_event_per_year,fullrange): # Computes the entropy of the distribution of the positive events over each century
    number_per_century=np.sum(mixed_event_per_year[fullrange].reshape((10,-1)),1)
    norm_per_century=number_per_century/np.sum(number_per_century)
    entropy_per_iteration_prime=-np.sum(norm_per_century*np.log(norm_per_century))
    return entropy_per_iteration_prime, number_per_century, norm_per_century
def PermuteFullrange(fullrange, randrange1, randrange2):# Create a permuted sequences based on the input labels
    returnfullrange = fullrange.copy()
    returnfullrange[randrange1], returnfullrange[randrange2] = returnfullrange[randrange2], returnfullrange[randrange1]
    return returnfullrange

def standardize_dim_names(xa:xr.DataArray) -> xr.DataArray:
    '''
    Renames the coordinates of `xa` to oblige with standard:
    longitude: 'lon'
    latitude:  'lat'
    time:      'time'

    The renamed dataarray is then returned
    '''
    standard_names_to_variants = {
        'lon': ['longitude', 'Longitude'],
        'lat': ['latitude', 'Latitude'],
        'time': ['Time', 't', 'T']
    }
    renamings = {}
    for dim in xa.dims:
        if dim in standard_names_to_variants:
            continue
        for standard_dim, variants in standard_names_to_variants.items():
            if dim in variants:
                renamings[dim] = standard_dim
                break
    if renamings:
        xa = xa.rename(renamings)
    return xa
    
def discard_all_dimensions_but(xa:xr.DataArray, dims_to_keep:list):
    missing_dims = set(dims_to_keep) - set(xa.dims)
    if missing_dims:
        logger.warning(f'Asking to keep dimensions {missing_dims}, which are not present in DataArray, trying to standardize names')
        xa = standardize_dim_names(xa)
        missing_dims = set(dims_to_keep) - set(xa.dims)
        if missing_dims:
            logger.error(f'{missing_dims} are still missing from DataArray: expect errors')
    dims_to_drop = [dim for dim in xa.dims if dim not in dims_to_keep]
    xa = xa.isel({dim: 0 for dim in dims_to_drop})
    xa = xa.drop(dims_to_drop)
    return xa

# AL: These two functions maybe should also sort the latitudes? It doesn't seem necessary at the moment because we do .sel(field.lat) anyways... however it may be a weak point of the code for the future
def get_lsm(mylocal,Model, discretize=True):
    if Model == 'Plasim':
        lsm = xr.open_dataset(ut.first_valid_path(mylocal, 'Data_Plasim_inter/CONTROL_lsmask.nc')).lsm
    elif Model == 'CESM':
        lsm = xr.open_dataset(ut.first_valid_path(mylocal, 'Data_CESM/CAM_landmask.nc')).landmask
    elif Model.startswith('ERA'):
        lsm = xr.open_dataset(ut.first_valid_path(mylocal, 'Data_ERA5/land_sea_mask.nc')).lsm
    else:
        raise NotImplementedError()
    lsm = standardize_dim_names(lsm)
    lsm = discard_all_dimensions_but(lsm, ['lon', 'lat'])
    if discretize:
        return (lsm > 0.5).astype(lsm.dtype)
    return lsm

def get_cell_area(mylocal,Model):
    if Model == 'Plasim':
        cell_area = xr.open_dataset(ut.first_valid_path(mylocal, 'Data_Plasim_inter/CONTROL_gparea.nc')).cell_area
    elif Model == 'CESM':
        cell_area = xr.open_dataset(ut.first_valid_path(mylocal, 'Data_CESM/CAM_cellarea.nc')).cell_area
    elif Model.startswith('ERA'):
        cell_area = xr.open_dataset(ut.first_valid_path(mylocal, 'Data_ERA5/cell_area.nc')).cell_area
    else:
        raise NotImplementedError()
    cell_area = standardize_dim_names(cell_area)
    cell_area = discard_all_dimensions_but(cell_area, ['lon', 'lat'])
    return cell_area

def ExtractAreaWithMask(mylocal,Model,area): # extract land sea mask and multiply it by cell area
    # Load the land area mask
    dataset = Dataset(ut.first_valid_path(mylocal,'Data_Plasim_inter/CONTROL_lsmask.nc'))
    lsm = dataset.variables['lsm'][0]
    dataset.close()
    # Load the areas of each cell
    dataset = Dataset(ut.first_valid_path(mylocal,'Data_Plasim_inter/CONTROL_gparea.nc'))
    cell_area = dataset.variables["cell_area"][:]
    dataset.close()
    # mask_ocean = np.array(lsm) # unused
    # mask_area = np.array(create_mask(Model,area, lsm)) # unused


    mask = create_mask(Model,area,cell_area)*create_mask(Model,area,lsm)
    mask = mask/np.sum(mask)  # Here I combine both grid-point area times the mask into a normalized mask
    return mask, cell_area, lsm

def TryLocalSource(mylocal):
    folder = mylocal
    addresswithoutlocal = folder[7:]
    logger.info(f"{addresswithoutlocal = }")
    mylocal = '/ClimateDynamics/MediumSpace/ClimateLearningFR/' # This is a hard overwrite to prevent looking in other folders and slow down say scratch. If something doesn't work in backward compatibility, remove it
    folder = mylocal+addresswithoutlocal
    logger.info(f"Trying source: {mylocal}") # We assume the input has the form: '/local/gmiloshe/PLASIM/'+''+'Data_Plasim/'
    if not os.path.exists(folder):
        folder='/projects/users/'+addresswithoutlocal
        logger.info(f"Trying source: {folder}")
        if not os.path.exists(folder):
            folder='/ClimateDynamics/MediumSpace/ClimateLearningFR/'+addresswithoutlocal
            logger.info(f"Trying source: {folder}")
            if not os.path.exists(folder):
                folder='/scratch/'+addresswithoutlocal
                logger.info(f"Trying source: {folder}")
                if not os.path.exists(folder):
                    logger.warning("Input data could not be found")
    logger.info(f"The source will be: {folder}")
    return folder

def SingleLinearRegression(i,X, y):
    # Split the data into training/testing sets
    a = i*X.shape[0]//10
    b = (i+1)*X.shape[0]//10
    X_train = np.concatenate((X[:a],X[b:]))
    X_test = X[a:b]
    #print("X_train.shape = ",X_train.shape, ", X_test.shape = ",X_test.shape, " , a = ", a, " , b = ", b)

    # Split the targets into training/testing sets
    y_train = np.concatenate((y[:a],y[b:]))
    y_test = y[a:b]
    #print('y_train.shape = ',y_train.shape, ", y_test.shape = ",y_test.shape)
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)
    
    R2 = r2_score(y_test, y_pred)
    #print(y_pred.shape)
    return X_test, X_train, y_test, y_train, regr, y_pred, R2

def TrainTestSplit(i,X, labels, undersampling_factor,verbose): # OLD VERSION # CAN SOON BE PHASED OUT
    # Split the data into training/testing sets
    a = i*X.shape[0]//10
    b = (i+1)*X.shape[0]//10
    X_train = np.concatenate((X[:a],X[b:])) 
    X_test = X[a:b]
    if verbose:
        print("i: ", i, " , X_train.shape = ", X_train.shape, ", X_test.shape = ", X_test.shape, " , a = ", a, " , b = ", b)
    
    # Split the targets into training/testing sets
    Y_train = labels[:a]+labels[b:] # a true/false list, true for a heatwave
    Y_test = labels[a:b]
    if verbose:
        print("Y_train positives = ",np.sum(Y_train), "Y_train negatives = ",np.sum(list(~np.array(Y_train))), " , Y_test positives = ", np.sum(Y_test), " , Y_test negatives = ", np.sum(list(~np.array(Y_test))))
    Y_train_positives = list(True for iterate in range(np.sum(Y_train)))
    Y_train_negatives = list(False for iterate in range(np.sum(list(~np.array(Y_train)))))
    X_train_positives = X_train[Y_train]
    X_train_negatives = X_train[list(~np.array(Y_train))]
    
    filename = 'Postproc/permutation_'+str(i)+'_undersampling_'+str(undersampling_factor)+'.npy'
    if os.path.exists(filename):
        print('loading '+filename)
        permutation_negatives = np.load(filename)
    else: 
        permutation_negatives = np.random.permutation(len(Y_train_negatives)) #creates a permuation of integers up to len(Y_train_negatives)
        print('saving '+filename)
        np.save(filename, permutation_negatives)
                                                                    
    X_train_negatives_permuted = X_train_negatives[permutation_negatives,:] # permuted original series
    X_train_negatives_truncated = X_train_negatives_permuted[:len(Y_train_negatives)//undersampling_factor,:] # select a subset that is undersampled
    #if verbose:
        #logger.info("len(Y_train_negatives)//undersampling_factor = ",len(Y_train_negatives)//undersampling_factor)
    Y_train_negatives_truncated = list(False for iterate in range(len(Y_train_negatives)//undersampling_factor)) # create mock labels
    #if verbose:
        #print(X_train_negatives_truncated.shape,len(Y_train_negatives_truncated))
    Y_train_new = Y_train_positives+Y_train_negatives_truncated
    X_train_new = np.concatenate((X_train_positives,X_train_negatives_truncated))
    if verbose:
        print("Y_train positives = ",np.sum(Y_train_new), "Y_train negatives = ",len(Y_train_new) - np.sum(Y_train_new)) # now labels are 0 or 1 so we can't simply do:  np.sum(list(~np.array(Y_train_new)))
        
    return X_train, X_test, Y_train, Y_test, X_train_new, Y_train_new


def TrainTestSplitIndices(i,X, labels, undersampling_factor, sampling='', newundersampling=False, thefield='', percent=5, contain_folder = 'Postproc', num_batches=10, j=1): # returns indices of the test labels, and train labels (undersampled or not)
    # Split the data into training/testing sets
    # i is the beginning, j corresponds to the end
    lower = int(i*X.shape[0]//num_batches)   # select the lower bound
    upper = int((i+j)*X.shape[0]//num_batches) # select the upper bound
    logger.info(f"initial: {lower = }, {upper = }")
    if upper<= X.shape[0]: # The usual situation we encounter 
        test_indices = np.array(range(lower,upper))  # extract the test set which is between the lower and the upper bound
        # next we select the train set which is below the lower bound and above the uppder bound
        train_indices = np.array(list(range(lower))+list(range(upper,X.shape[0])))  # The indices of the train set (relative to the original set)
    else: # This happens if we chose large test sets and leave-one-out algorithm surpasses the size of our data, the rest of the test set starts from the beginning of the dataset
        upper = upper - X.shape[0]
        logger.info(f"upper bound changed to {upper}")
        train_indices = np.array(range(upper,lower))  # extract the train set which is between the lower and the upper bound
        # next we select the test set which is below the lower bound and above the uppder bound
        test_indices = np.array(list(range(upper))+list(range(lower,X.shape[0])))  # The indices of the train set (relative to the original set)

    train_labels_indices = (labels[train_indices])                               # Array of labels of the train set (relative to the train set)
    train_true_labels_indices = (train_indices[(train_labels_indices)])               # The array of the indices of the true labels in the train set
    train_false_labels_indices = (train_indices[(~train_labels_indices)])             # The array indices of the false labels in the train set
    logger.info(f"{train_false_labels_indices.shape[0] = }")
    if undersampling_factor > 1: # if we need to manually remove some positive labels # THIS PART OF THE CODE SHOULD BE PHASED OUT NOW THAT WE HAVEN'T USED IT FOR A WHILE
        ## The old version for compatibility:
        #filename = 'Postproc/permutation_'+str(i)+'_sampling_'+str(sampling)+'_'+thefield+'.npy'
        ## The new version:
        filename = contain_folder+'/permutation_'+str(i)+'_sampling_'+str(sampling)+'_'+thefield+'_per_'+str(percent)+'.npy'
        if (not os.path.exists(filename)) or newundersampling:
            permutation_negatives = np.random.permutation(len(train_false_labels_indices)) #creates a permuation of integers up to len(train_false_labels_indices)
            logger.info(f'saving {filename}')
            np.save(filename, permutation_negatives)
        else: 
            logger.info(f'loading {filename}')
            permutation_negatives = np.load(filename)
        
        train_false_labels_undersampled_indices = train_false_labels_indices[permutation_negatives[:train_false_labels_indices.shape[0]//undersampling_factor]]
        train_indices = np.concatenate((train_true_labels_indices,train_false_labels_undersampled_indices))
    else:
        filename = []
    return test_indices, train_indices, train_true_labels_indices, train_false_labels_indices, filename

def TrainTestSampleIndices(i,Xshape, labels, undersampling_factor, sampling='', newundersampling=False, thefield=''): # returns indices of the test/train split while sampling randomly from the full set
    # Split the data into training/testing sets
    lower = i*Xshape[0]//10   # select the lower bound
    upper = (i+1)*Xshape[0]//10 # select the upper bound

    test_indices = np.array(range(lower,upper))  # extract the test set which is between the lower and the upper bound
    # next we select the train set which is below the lower bound and above the uppder bound
    train_indices = np.array(list(range(lower))+list(range(upper,Xshape[0])))  # The indices of the train set (relative to the original set)
    train_labels_indices = (labels[train_indices])                               # Array of labels of the train set (relative to the train set)
    train_true_labels_indices = (train_indices[(train_labels_indices)])               # The array of the indices of the true labels in the train set
    train_false_labels_indices = (train_indices[(~train_labels_indices)])             # The array indices of the false labels in the train set

    if undersampling_factor > 1: # if we need to manually remove some positive labels
        filename = 'Postproc/permutation_'+str(i)+'_sampling_'+str(sampling)+'_'+thefield+'.npy'
        if ((not os.path.exists(filename)) or newundersampling):
            permutation_negatives = np.random.permutation(len(train_false_labels_indices)) #creates a permuation of integers up to len(Y_train_negatives)
            logger.info(f'saving {filename}')
            np.save(filename, permutation_negatives)
        else:
            logger.info(f'loading {filename}')
            permutation_negatives = np.load(filename)
        # First undersample false labels randomly
        train_false_labels_undersampled_indices = train_false_labels_indices[permutation_negatives[:train_false_labels_indices.shape[0]//undersampling_factor]] 
        train_indices = shuffle(np.concatenate((train_true_labels_indices,train_false_labels_undersampled_indices))) # next shuffle it with positive labels
    return test_indices, train_indices, train_true_labels_indices, train_false_labels_indices


def TrainTestSplit2(i,X, labels, undersampling_factor,verbose): # NEW VERSION (removes the need for extra ouputs and extra shuffling if oversampling == 1)
    # Split the data into training/testing sets  # we can retire this function
    a = i*X.shape[0]//10
    b = (i+1)*X.shape[0]//10
    X_train = np.concatenate((X[:a],X[b:])) 
    X_test = X[a:b]
    if verbose:
        print("i: ", i, " , X_train.shape = ", X_train.shape, ", X_test.shape = ", X_test.shape, " , a = ", a, " , b = ", b)
    
    # Split the targets into training/testing sets
    Y_train = labels[:a]+labels[b:] # a true/false list, true for a heatwave
    Y_test = labels[a:b]
    if verbose:
        print("Y_train positives = ",np.sum(Y_train), "Y_train negatives = ",np.sum(list(~np.array(Y_train))), " , Y_test positives = ", np.sum(Y_test), " , Y_test negatives = ", np.sum(list(~np.array(Y_test))))
    if undersampling_factor > 1: # if we need to manually remove some positive labels
        Y_train_positives = list(True for iterate in range(np.sum(Y_train)))
        Y_train_negatives = list(False for iterate in range(np.sum(list(~np.array(Y_train)))))
        X_train_positives = X_train[Y_train]
        X_train_negatives = X_train[list(~np.array(Y_train))]
    
        filename = 'Postproc/permutation_'+str(i)+'_undersampling_'+str(undersampling_factor)+'.npy'
        if os.path.exists(filename):
            logger.info(f'loading {filename}')
            permutation_negatives = np.load(filename)
        else: 
            permutation_negatives = np.random.permutation(len(Y_train_negatives)) #creates a permuation of integers up to len(Y_train_negatives)
            logger.info(f'saving {filename}')
            np.save(filename, permutation_negatives)
                                                                    
        X_train_negatives_permuted = X_train_negatives[permutation_negatives,:] # permuted original series
        X_train_negatives_truncated = X_train_negatives_permuted[:len(Y_train_negatives)//undersampling_factor,:] # select a subset that is undersampled
        #if verbose:
            #print("len(Y_train_negatives)//undersampling_factor = ",len(Y_train_negatives)//undersampling_factor)
        Y_train_negatives_truncated = list(False for iterate in range(len(Y_train_negatives)//undersampling_factor)) # create mock labels
        #if verbose:
            #print(X_train_negatives_truncated.shape,len(Y_train_negatives_truncated))
        Y_train_new = Y_train_positives+Y_train_negatives_truncated
        X_train_new = np.concatenate((X_train_positives,X_train_negatives_truncated))
    else:
        Y_train_new = Y_train
        X_train_new = X_train
    if verbose:
        print("Y_train positives = ",np.sum(Y_train_new), "Y_train negatives = ",len(Y_train_new) - np.sum(Y_train_new)) # now labels are 0 or 1 so we can't simply do:  np.sum(list(~np.array(Y_train_new)))
        
    return X_test, Y_test, X_train_new, Y_train_new

def PrepareTestTrain2(X_train_new, X_test): # new version. Get's our data ready for training (reducing size and normalization)
    std_X_train_new = np.std(X_train_new,0) # Normalization
    mean_X_train_new = np.mean(X_train_new,0)
    std_X_train_new[std_X_train_new==0] = 1 # If there is no variance we shouldn't divide by zero

    X_train_new = ((X_train_new-mean_X_train_new)/std_X_train_new) 
    X_test = ((X_test-mean_X_train_new)/std_X_train_new) 

def PrepareTestTrain(X_train_new, X_test, Y_train_new,Y_test): # Get's our data ready for training (reducing size and normalization)
    #Y_test = np.array(Y_test, np.uint8)
    #Y_train_new = np.array(Y_train_new, np.uint8)
    
    Y_test = np.array(Y_test)
    Y_train_new = np.array(Y_train_new)

    std_X_train_new = np.std(X_train_new,0) # Normalization
    mean_X_train_new = np.mean(X_train_new,0)
    std_X_train_new[std_X_train_new==0] = 1 # If there is no variance we shouldn't divide by zero

    X_train_new = ((X_train_new-mean_X_train_new)/std_X_train_new) 
    X_test = ((X_test-mean_X_train_new)/std_X_train_new) 

    return X_train_new, X_test, Y_train_new, Y_test

def ComputeMCC(Y_test, Y_pred, verbose=False):
    # Compute Matthews Correlation Coefficient
    [[TN, FP],[FN, TP]] = confusion_matrix(Y_test, Y_pred) # note that confusion matrix treats 0 as the first column/row
    [[TNd, FPd],[FNd, TPd]] = np.array([[TN, FP],[FN, TP]])
    if ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) == 0):
        MCC = 0
    else:
        MCC = ((TPd * TNd - FPd *FNd)/ np.sqrt(float((TPd+FPd)*(TPd+FNd)*(TNd+FPd)*(TNd+FNd))))
    if verbose:
        print(f'{MCC = }, {TP = }, {TN = }, {FP = }, {FN = }')
    logger.info(f'{MCC = }, {TP = }, {TN = }, {FP = }, {FN = }')
    return TP, TN, FP, FN, MCC
        
def SingleLogisticRegression(i,X, labels, undersampling_factor,verbose,inv_reg=1e5):
    # Perform standard Logistic Regression
    X_train, X_test, Y_train, Y_test, X_train_new, Y_train_new = TrainTestSplit(i,X, labels, undersampling_factor,verbose)# Split the data into training/testing sets
    
    logreg = LogisticRegression(solver='liblinear',C=inv_reg)
    logreg.fit(X_train_new, Y_train_new)
    Y_pred = logreg.predict(X_test) # confusion matrix works despite the fact that Y_test is True/False and Y_pred is 1/0
    
    TP, TN, FP, FN, MCC = ComputeMCC(Y_test, Y_pred, verbose)
    
    return X_test, X_train, Y_test, Y_train, logreg, Y_pred, TP, TN, FP, FN, MCC

def PlotLinearRegression(Xname,R2,ax, X_test, y_test, y_pred):
    #plt.rcParams['pcolor.shading'] ='flat'  # This parameter doesn't work with my version of python
    myhist = ax.hist2d(X_test[:,0], y_test)#, bins=20, cmap='Blues')
    cb = plt.colorbar(myhist[3], ax = ax)
    #plt.scatter(X_test, y_test,  color='black')
    ax.plot(X_test, y_pred, color='red', linewidth=3)
    ax.set_xlabel(Xname)
    ax.set_ylabel('Runnin mean temperature')
    ax.set_title('Coef. of determ. %.2f' % (R2))

def PlotLogisticRegression(X,Xname,logreg,ax, X_test, Y_test, TP, TN, FP, FN, MCC):
    xx = np.linspace(X[:, 0].min(), X[:, 0].max(),1000)
    Z = logreg.predict(np.c_[xx])
    # Put the result into a color plot
    myhist = ax.hist(X_test[Y_test], 20, density = True, alpha = 0.75, label ='True label')
    myhist = ax.hist(X_test[list(~np.array(Y_test))], 20, density = True, alpha = 0.75, label = 'False label')
    ax.set_title('MCC = %.2f, TP =  %d, TN =  %d, FP =  %d, FN =  %d' % (MCC,TP,TN,FP,FN))
    ax.set_xlabel(Xname)
    ax.axvline(xx[list(np.abs(np.diff(Z))).index(1)],color = 'black')
    ax.text(xx[list(np.abs(np.diff(Z))).index(1)] + 0.005, 0, '{:.4f}'.format(xx[list(np.abs(np.diff(Z))).index(1)]))
    ax.set_xlim([xx.min(),xx.max()])
    ax.legend(loc = 'best')
    
def Plot2DLogisticRegression(X,Xname,logreg,ax, X_test, Y_test, TP, TN, FP, FN, MCC):
    plt.rcParams['pcolor.shading'] ='gouraud'
    xx = np.linspace(X[:, 0].min(), X[:, 0].max(),1000)
    yy = np.linspace(X[:, 1].min(), X[:, 1].max(),1000)
    XX, YY = np.meshgrid(xx, yy)
    Z = logreg.predict(np.c_[XX.ravel(), YY.ravel()])
    # Put the result into a color plot
    
    Z = Z.reshape(XX.shape)
    ax.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired,alpha=0.5)

    # Put the result into a color plot
    ax.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, edgecolors='k', cmap=plt.cm.Paired,alpha=0.35)
    ax.set_xlim([xx.min(),xx.max()])
    ax.set_ylim([yy.min(),yy.max()])
    ax.set_title('MCC = %.2f, TP =  %d, TN =  %d, FP =  %d, FN =  %d' % (MCC,TP,TN,FP,FN),  fontsize=10)
    ax.set_xlabel(Xname[0])
    ax.set_ylabel(Xname[1])
    
def ShowArea(LON_mask, LAT_mask, MyArea, coords=[-7,15,40,60], **kwargs):
    '''
    Show the area based on grid points enclosed in LON_mask LAT_mask
    
    `coords` is ignored in the Basemap version, showing only the region over France
    **kwargs are passed to cartopy_plots.ShowArea
    '''
    
    if plotter == 'cartopy':
        return cplt.ShowArea(LON_mask, LAT_mask, MyArea, coords, **kwargs)
    
    plt.rcParams['pcolor.shading'] ='flat'
    coords = [50, 5., 2000000, 2000000]
    fig = plt.figure(figsize=(15, 15), edgecolor='w')
    m = Basemap(width=coords[2],height=coords[3],resolution='l',projection='aea',lat_ts=60,lat_0=coords[0],lon_0=coords[1])
    m.pcolormesh((LON_mask), (LAT_mask), MyArea, latlon=True, alpha = .35, cmap='RdBu_r')
    plt.title("Area of a cell")
    m.colorbar()
    m.drawcoastlines()


class Logger(object): # This object is used to keep track of the output generated by print
    def __init__(self, address, logfile_name='logfile.log'):
        self.terminal = sys.stdout
        address = address.rstrip('/')
        self.log = open(f'{address}/{logfile_name}', "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def ReadStringFromFileRaw(filename, string2, verbose=True):
# Read the original py file that launched the training and find the relevant parameter without taking float part
    file1 = open(filename, "r")
    flag = 0
    index = 0
    parameter = []
    for line in file1:
        index += 1

        ## checking string is present in line or not
        if string2 in line:
        ## checking if the line begins with string2
        #if line[:len(string2)] == string2:
            line_index = line.find('= ')
            if line_index == -1: # if couldn't find
                line_index = line.find('=')
            line_index2 = line.find('#')
            if verbose:
                print("found =space at", line_index, " in |", line, "| with length = ", len(line), " extracting |", line[line_index+1:line_index2], "|")
            parameter = (line[line_index+1:line_index2])#-1])
            flag = 1
            if verbose:
                print("parameter = ", parameter, " at index = ", index)
                #print(string2+" = ", parameter, " ,index = ", index, " ,line = ", line)
            break
    if verbose:
        print(string2+" index = ", index)
    file1.close()
    return parameter


def ReadStringFromFile(file1, string2):
# Read the original py file that launched the training and find the relevant parameter
    flag = 0
    index = 0
    parameter = []
    for line in file1:
        index += 1

        ## checking string is present in line or not
        #if string2 in line:
        ## checking if the line begins with string2
        if line[:len(string2)] == string2:
          line_index = line.find('= ')
          if line_index == len(line): # if couldn't find
            line_index = line.find('=')
          line_index2 = line.find('#')
          
          print("found =space at", line_index, " in |", line, "| with length = ", len(line), " extracting |", line[line_index+1:line_index2], "|")
          parameter = float(line[line_index+1:line_index2])
          flag = 1
          print("parameter = ", parameter, " at index = ", index)
          #print(string2+" = ", parameter, " ,index = ", index, " ,line = ", line)
          break
    #print(string2+" index = ", index)
    return parameter
    

def ReNormProbability(Y_pred_prob_input, reundersampling_factor=1):
    '''
    Old version of unbias_probabilities, to be phased out
    '''
    Y_pred_prob = Y_pred_prob_input.copy()
    Denominator = 1 - (1 - reundersampling_factor)*Y_pred_prob[:,0]
    Y_pred_prob = np.c_[reundersampling_factor*Y_pred_prob[:,0]/Denominator, Y_pred_prob[:,1]/Denominator]
    Y_pred_prob_eq_0 = Y_pred_prob == 0
    Y_pred_prob[Y_pred_prob_eq_0] = 1e-15 #renormalize to machine precision when Y_pred_prob = 0, because it can cause problems when taking log
    Y_pred_prob_eq_0[:,[0,1]] = Y_pred_prob_eq_0[:,[1,0]]  # switch the columns so that the truth value now corresponds to the other column that is supposed to be set to 1 - 1e-15
    Y_pred_prob[Y_pred_prob_eq_0] = 1 - 1e-15 #renormalize when Y_pred_prob = 1
    return Y_pred_prob


def ComputeMetrics(Y_test, Y_pred_prob_input, percent, reundersampling_factor=1, print_output=True):  # Compute the metrics given the probabilities and the outcome
    """
    Denominator = 1 - (1 - reundersampling_factor)*Y_pred_prob[:,0]
    Y_pred_prob = np.c_[reundersampling_factor*Y_pred_prob[:,0]/Denominator, Y_pred_prob[:,1]/Denominator]
    Y_pred_prob_eq_0 = Y_pred_prob == 0
    Y_pred_prob[Y_pred_prob_eq_0] = 1e-15 #renormalize to machine precision when Y_pred_prob = 0, because it can cause problems when taking log
    Y_pred_prob_eq_0[:,[0,1]] = Y_pred_prob_eq_0[:,[1,0]]  # switch the columns so that the truth value now corresponds to the other column that is supposed to be set to 1 - 1e-15
    Y_pred_prob[Y_pred_prob_eq_0] = 1 - 1e-15 #renormalize when Y_pred_prob = 1
    """

    Y_pred_prob = ReNormProbability(Y_pred_prob_input, reundersampling_factor)
    label_assignment = np.argmax(Y_pred_prob,1)

    TP, TN, FP, FN, new_MCC = ComputeMCC(Y_test, label_assignment, verbose=True)
    new_entropy = -np.sum(np.c_[1-Y_test,Y_test]*np.log(Y_pred_prob))/Y_test.shape[0]
    #new_entropy[i] = -np.sum(Y_test_2_1hot*np.log(Y_pred_prob))
    maxskill = -(percent/100.)*np.log(percent/100.)-(1-percent/100.)*np.log(1-percent/100.)
    new_skill = (maxskill-new_entropy)/maxskill
    new_BS = np.sum((Y_test-Y_pred_prob[:,1])**2)/Y_test.shape[0]
    new_WBS =  np.sum((100./(100.-percent-(100.-2*percent)*Y_test))*(Y_test-Y_pred_prob[:,1])**2)/np.sum((100./(100.-percent-(100.-2*percent)*Y_test)))
    new_freq = np.sum(label_assignment)/Y_test.shape[0]
    if print_output:
        logger.info("renorm = ", reundersampling_factor,", MCC = " , new_MCC," ,entropy = ", new_entropy, " ,entropy = ", -np.sum(np.c_[1-Y_test,Y_test]*np.log(Y_pred_prob))/Y_test.shape[0], " , skill = ", new_skill," ,BS = ", new_BS, " , WBS = ", new_WBS, " , freq = ", new_freq)
    
    return new_MCC, new_entropy, new_skill, new_BS, new_WBS, new_freq

def NormalizeAndX_test(i, X, mylabels, undersampling_factor, sampling, new_mixing, thefield, percent, training_address) :
    # This function reads X_mean and X_std and normalizes X_test which is given as an output
    if isinstance(X, list): # In the new system X consists of lists (useful for fused or combined CNNs)
        test_indices, train_indices, train_true_labels_indices, train_false_labels_indices, filename_permutation = TrainTestSplitIndices(i,X[0], mylabels, undersampling_factor, sampling, new_mixing, thefield, percent, training_address)
        for Xdim in range(len(X)):
            print("dimension of the train set is X[", Xdim, "][train_indices].shape = ", X[Xdim][train_indices].shape)
        print("# of the true labels = ", np.sum(mylabels[train_indices]))
        print("effective sampling rate for rare events is ", np.sum(mylabels[train_indices])/X[0][train_indices].shape[0])

        X_mean = np.load(training_address+'/batch_'+str(i)+'_X_mean.npy',allow_pickle=True)
        X_std = np.load(training_address+'/batch_'+str(i)+'_X_std.npy',allow_pickle=True)
        X_test = []
        for iter in range(len(X_std)):
            X_test.append( (X[iter][test_indices]-X_mean[iter])/X_std[iter] )
    else:# In the old system X used to be an array (useful for stacked CNN)
        test_indices, train_indices, train_true_labels_indices, train_false_labels_indices, filename_permutation = TrainTestSplitIndices(i,X, mylabels, undersampling_factor, sampling, new_mixing, thefield, percent, training_address)
        X_mean = np.load(training_address+'/batch_'+str(i)+'_X_mean.npy')
        X_std = np.load(training_address+'/batch_'+str(i)+'_X_std.npy')
        X_test = (X[test_indices]-X_mean)/X_std
    Y_test = mylabels[test_indices]
    
    return X_test, Y_test, X_mean, X_std, test_indices
