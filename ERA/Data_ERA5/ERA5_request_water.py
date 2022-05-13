import sys
import cdsapi

c = cdsapi.Client()

year=sys.argv[1]
c.retrieve(
    #'reanalysis-era5-land',
    'reanalysis-era5-single-levels',
    #'reanalysis-era5-single-levels-preliminary-back-extension',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3',
        ],
        'year': year,
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
        'time': '12:00',
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'grid': "0.75/0.75",
    },
    'ERA5_water_'+year+'.nc')
