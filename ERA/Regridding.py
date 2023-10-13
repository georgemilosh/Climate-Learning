
import xarray as xr
import xesmf as xe

CONTROL_lsmask = xr.open_dataset('/homedata/gmiloshe/Climate-Learning/PLASIM/Data_Plasim_inter/CONTROL_lsmask.nc')


ds_out = xr.Dataset(
    {
        "lat": (["lat"], CONTROL_lsmask.lat.data),
        "lon": (["lon"], CONTROL_lsmask.lon.data),
    }
)
from_folder = '/net/nfs/ssd2/gmiloshe/Climate-Learning/ERA/ERA5/linear/'
to_folder = 'Data_ERA5_regridded/'
for field in ['ANO_t2m','ANO_zg500','ANO_water_weighted']:
    year_dataset = xr.open_dataset(f'{from_folder}{field}.nc')
    print(f'{field = } opened from {from_folder}{field}.nc')
    regridder = xe.Regridder(year_dataset, ds_out, "bilinear")
    print(f'{regridder = }')
    year_dataset_out = regridder(year_dataset)
    print(f'{field = } regridded')
    year_dataset_out.to_netcdf(f'{to_folder}{field}.nc')
    print(f'regridded {field = } saved to {to_folder}{field}.nc')
