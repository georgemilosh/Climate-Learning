# This routine regrids the CESM output to Plasim grid
import xarray as xr
import xesmf as xe

CONTROL_lsmask = xr.open_dataset('/homedata/gmiloshe/Climate-Learning/PLASIM/Data_Plasim_inter/CONTROL_lsmask.nc')


ds_out = xr.Dataset(
    {
        "lat": (["lat"], CONTROL_lsmask.lat.data),
        "lon": (["lon"], CONTROL_lsmask.lon.data),
    }
)

field = 'zg500'
from_folder = '/net/nfs/ssd2/gmiloshe/Climate-Learning/ERA/Data_ERA5/ERA5'
to_folder = 'Data_ERA5_regridded/ERA5'

for year in range(1950,2020):
    year_dataset = xr.open_dataset(f'{from_folder}_{field}_{year}.nc')
    print(f'{year = } opened from {from_folder}_{field}_{year}.nc')
    regridder = xe.Regridder(year_dataset, ds_out, "bilinear")
    print(f'{regridder = }')
    year_dataset_out = regridder(year_dataset)
    print(f'{year = } regridded')
    year_dataset_out.to_netcdf(f'{to_folder}_{field}_{year}.nc')
    print(f'regridded {year = } saved to {to_folder}_{field}_{year}.nc')
