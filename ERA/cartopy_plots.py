import cartopy.crs as ccrs
import cartopy.feature as cfeat
data_proj = ccrs.PlateCarree()

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