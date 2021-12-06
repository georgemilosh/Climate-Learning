import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.util import add_cyclic_point as acp

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

import numpy as np
import warnings

import ERA_Fields as ef

data_proj = ccrs.PlateCarree()

def Greenwich(*args):
    '''
    Adds a new copy of the Greenwich meridian at the end of a series of arrays. Useful for plotting data around the pole.
    
    USAGE:
        extended_array = Greenwich(array)
    or
        extended_lon, *list_of_extended_arrays = Greenwich(lon, *list_of_arrays)
    
    If a single argument is provided, the first 'column' is copied to the end
    
    If more arguments are provided the first one is assumed to be longitude data, for which the added column will be filled with the value 360
    '''
    if len(args) == 1:
        return acp(args[0])
    args = [acp(a) for a in args]
    args[0][...,-1] = 360 # fix the longitude
    return args

def draw_map(m, background='stock_img', **kwargs):
    '''
    Plots a background map using cartopy.
    Additional arguments are passed to the cartopy function gridlines
    
    Parameters:
    -----------
        m: cartopy axis
        resolution: either 'low' or 'high'
        **kwargs: arguments passed to cartopy.gridlines
    '''
    if 'draw_labels' not in kwargs:
        kwargs['draw_labels'] = True

    if background == 'stock_img':
        m.stock_img()
    elif background == 'land-sea':
        m.add_feature(cfeat.LAND)
        m.add_feature(cfeat.OCEAN)
        m.add_feature(cfeat.LAKES)
    else:
        if background != 'coastlines':
            warning.warn(f"Unrecognized option {background = }, using 'coastlines' instead")
        m.coastlines()
    m.gridlines(**kwargs)
    
def geo_contourf(m, lon, lat, values, levels=None, cmap='RdBu_r', title=None, put_colorbar=True):
    '''
    Contourf plot together with coastlines and meridians
    
    Parameters:
    -----------
        m: cartopy axis
        lon: 2D longidute array
        lat: 2D latitude array
        values: 2D field array
        levels: contour levels for the field values
        cmap: colormap
        title: plot title
        put_colorbar: whether to show a colorbar
    '''
    im = m.contourf(lon, lat, values, transform=data_proj,
                    levels=levels, cmap=cmap, extend='both')
    m.coastlines()
    m.gridlines(draw_labels=True)
    if put_colorbar:
        plt.colorbar(im)
    if title is not None:
        m.set_title(title, fontsize=20)
        
def geo_contour(m, lon, lat, values, levels=None, cmap1='PuRd', cmap2=None):
    '''
    Plots a contour plot with the possbility of having two different colormaps for positive and negative data
    
    Parameters:
    -----------
        m: cartopy axis
        lon: 2D longidute array
        lat: 2D latitude array
        values: 2D field array
        levels: contour levels for the field values
        cmap1: principal colormap
        cmap2: if provided negative values will be plotted with `cmap1` and positive ones with `cmap2`
    '''
    if cmap2 is None: # plot with just one colormap
        m.contour(lon, lat, values, transform=data_proj,
                  levels=levels, cmap=cmap1)
    else: # separate positive and negative data
        v_neg = values.copy()
        v_neg[v_neg > 0] = 0
        m.contour(lon, lat, v_neg, transform=data_proj,
                  levels=levels, cmap=cmap1, vmin=levels[0], vmax=0)
        v_pos = values.copy()
        v_pos[v_pos < 0] = 0
        m.contour(lon, lat, v_pos, transform=data_proj,
                  levels=levels, cmap=cmap2, vmin=0, vmax=levels[-1])
        
def geo_contour_color(m, lon, lat, values, t_values, t_threshold, levels,
                      colors=["sienna","chocolate","green","lime"], linestyles=["solid","dashed","dashed","solid"],
                      linewidths=[1,1,1,1], fmt='%1.0f', fontsize=12):
    '''
    Plots contour lines divided in four categories: in order
        significative negative data
        non-significative negative data
        non-significative positive data
        significative positive data
        
    Significance is determined by comparing the `t_values`, which is an array of the same shape of `lon`, `lat' and `values`, with `t_threshold`
    
    Parameters:
    -----------
        m: cartopy axis
        lon: 2D longidute array
        lat: 2D latitude array
        values: 2D field array
        t_values: 2D array of the t_field (significance)
        t_threshold: float, t values above the threshold are considered significant
        levels: contour levels for the field values
        
        fmt: fmt of the inline contour labels
        fontsize: fontsize of the inline contour labels
        
    For the following see above for the order of the items in the lists
        colors
        linestyles
        linewidths
    '''
    
    # divide data in significative and non significative:
    data_sig, _ = ef.significative_data(values, t_values, t_threshold, both=False, default_value=np.NaN)
    
    # negative insignificant anomalies
    i = 1
    v_neg = values.copy()
    v_neg[v_neg > 0] = 0
    cn = m.contour(lon, lat, v_neg, transform=data_proj,
                   levels=levels, colors=colors[i], linestyles=linestyles[i], linewidths=linewidths[i])
    m.clabel(cn, colors=[colors[i]], manual=False, inline=True, fmt=fmt, fontsize=fontsize)
    # positive insignificant anomalies
    i = 2
    v_pos = values.copy()
    v_pos[v_pos < 0] = 0
    cp = m.contour(lon, lat, v_pos, transform=data_proj,
                   levels=levels, colors=colors[i], linestyles=linestyles[i], linewidths=linewidths[i])
    m.clabel(cp, colors=[colors[i]], manual=False, inline=True, fmt=fmt, fontsize=fontsize)
    
    # negative significant anomalies
    i = 0
    v_neg = data_sig.copy()
    v_neg[v_neg > 0] = 0
    cn = m.contour(lon, lat, v_neg, transform=data_proj,
                   levels=levels, colors=colors[i], linestyles=linestyles[i], linewidths=linewidths[i])
    # m.clabel(cn, colors=[colors[i]], manual=False, inline=True, fmt=fmt, fontsize=fontsize)
    # positive significant anomalies
    i = 3
    v_pos = data_sig.copy()
    v_pos[v_pos < 0] = 0
    cp = m.contour(lon, lat, v_pos, transform=data_proj,
                   levels=levels, colors=colors[i], linestyles=linestyles[i], linewidths=linewidths[i])
    # m.clabel(cp, colors=[colors[i]], manual=False, inline=True, fmt=fmt, fontsize=fontsize)
        
def PltMaxMinValue(m, lon, lat, values, colors=['red','blue']):
    '''
    Writes on the plot the maximum and minimum values of a field.
    
    Parameters:
    -----------
        m: cartopy axis
        lon: 2D longidute array
        lat: 2D latitude array
        values: 2D field array
        colors: the two colors of the text, respectively for the min and max values
    '''
    # plot min value
    coordsmax = tuple(np.unravel_index(np.argmin(values, axis=None), values.shape))
    x, y = lon[coordsmax], lat[coordsmax]
    txt = m.text(x, y, f"{np.min(values) :.0f}", transform=data_proj, color=colors[0])
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
    # plot max value
    coordsmax = tuple(np.unravel_index(np.argmax(values, axis=None), values.shape))
    x, y = lon[coordsmax], lat[coordsmax]
    txt = plt.text(x, y, f"{np.max(values) :.0f}", transform=data_proj, color=colors[1])
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
        
        

def ShowArea(lon_mask, lat_mask, field_mask, coords=[-7,15,40,60], **kwargs):
    '''
    Shows the grid points, colored with respect to a given field, for instance the area of the cell
    
    Parameters:
    ----------- 
        lon_mask: 2D array of longitude grid points
        lat_mask: 2D array of same shape as lon_mask with the latitudes
        field_mask: 2D field (e.g. area of the grid cells) array of same shape as lon_mask
        coords: limits of the plot in the format [min_lon, max_lon, min_lat, max_lat]

        **kwargs:
            projection: default ccrs.PlateCarree()
            background: 'coastlines' (default), 'stock_img' or 'land-sea'
            figsize: default (15,15)
            draw_labels: whether to show lat and lon labels, default True
            show_grid: whether to display the grid connecting data points, default True
            title: default 'Area of a grid cell'
    '''
    # extract additional arguments
    projection = kwargs.pop('projection', ccrs.PlateCarree())
    background = kwargs.pop('background', 'coastlines')
    figsize = kwargs.pop('figsize', (15,15))
    draw_labels = kwargs.pop('draw_labels', True)
    show_grid = kwargs.pop('show_grid', True)
    title = kwargs.pop('title', 'Area of a grid cell')
    
    fig = plt.figure(figsize=figsize)
    m = plt.axes(projection=projection)
    m.set_extent(coords, crs=ccrs.PlateCarree())
    
    draw_map(m, background, draw_labels=draw_labels)
    
    if show_grid:
        # make longitude monotonically increasing
        _lon_mask = lon_mask.copy()
        modify = False
        for i in range(lon_mask.shape[1] - 1):
            if lon_mask[0,i] > lon_mask[0,i+1]:
                modify = True
                break
        if modify:
            _lon_mask[:,:i+1] -= 360
        # print(_lon_mask)
        
        m.pcolormesh(_lon_mask, lat_mask, np.ones_like(lon_mask), transform=data_proj,
                     alpha=0.35, cmap='Greys', edgecolors='grey')
    
    im = m.scatter(lon_mask, lat_mask, c=field_mask, transform=data_proj,
                   s=500, alpha = .35, cmap='RdBu_r')
    plt.title(title)
    plt.colorbar(im)