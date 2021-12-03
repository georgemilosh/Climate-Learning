import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.util import add_cyclic_point as acp

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

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

def draw_map(m, resolution='low', **kwargs):
    '''
    Plots a background map using cartopy.
    Additional arguments are passed to the cartopy function gridlines
    '''
    if 'draw_labels' not in kwargs:
        kwargs['draw_labels'] = True

    if resolution == 'low':
        m.stock_img()
    elif resolution == 'high':
        m.add_feature(cfeat.LAND)
        m.add_feature(cfeat.OCEAN)
        m.add_feature(cfeat.LAKES)
    
    m.gridlines(**kwargs)
    
def geo_contourf(m, lon, lat, values, levels=None, cmap='RdBu_r', title=None, put_colorbar=True):
    '''
    Contourf plot together with coastlines and meridians
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
                      linewidths=[1,1,1,1], fmt='%1.0f'):
    return
        
        
def PltMaxMinValue(m, lon, lat, values):
    # plot min value
    coordsmax = tuple(np.unravel_index(np.argmin(values, axis=None), values.shape))
    x, y = lon[coordsmax], lat[coordsmax]
    txt = m.text(x, y, f"{np.min(values) :.0f}", transform=data_proj, color='red')
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
    # plot max value
    coordsmax = tuple(np.unravel_index(np.argmax(values, axis=None), values.shape))
    x, y = lon[coordsmax], lat[coordsmax]
    txt = plt.text(x, y, f"{np.max(values) :.0f}", transform=data_proj, color='blue')
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
        
    