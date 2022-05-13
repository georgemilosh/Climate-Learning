import sys
import cdsapi

#            '2m_temperature', 'sea_ice_cover', 'sea_surface_temperature',
#            'skin_temperature',

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': 'Land-sea mask',
        'year': '1979',
        'month': '01',
        'day': '01',
        'time': '00:00',
        'grid': "0.75/0.75",
        'format': 'netcdf',
    },
    'Land-sea_mask.nc')

