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

from_folder = '/homedata/gmiloshe/Climate-Learning/CESM/Data_CESM/North_Anomalies_'
to_folder = 'Data_CESM_regridded/North_Anomalies_'

for field in ['TSA','H2OSOI','Z3.500hPa']:
    field_dataset = xr.open_dataset(f'{from_folder}{field}.nc')
    print(f'{field = } opened from {from_folder}{field}.nc')
    regridder = xe.Regridder(field_dataset, ds_out, "bilinear")
    print(f'{regridder = }')
    field_dataset_out = regridder(field_dataset)
    print(f'{field = } regridded')
    field_dataset_out.to_netcdf(f'{to_folder}{field}.nc')
    print(f'regridded {field = } saved to {to_folder}{field}.nc')
