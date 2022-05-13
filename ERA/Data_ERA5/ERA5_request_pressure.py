import sys
import cdsapi

#            '2m_temperature', 'sea_ice_cover', 'sea_surface_temperature',
#            'skin_temperature',
variable=sys.argv[1]
year=sys.argv[2]

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels',
    #'reanalysis-era5-pressure-levels-preliminary-back-extension',
    {
        'product_type': 'reanalysis',
        'variable': variable,
        'pressure_level': '500',
        'year': year,
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', '03:00', '06:00',
            '09:00', '12:00', '15:00',
            '18:00', '21:00',
        ],
        'grid': "0.75/0.75",
        'format': 'netcdf',
    },
    'download_zg500.nc')

