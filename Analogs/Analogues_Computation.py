import datetime as dt  # Python standard library datetime  module
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def ncdump(nc_fid, verb=True):
    def print_ncattr(key):
        try:
            print ('\t\ttype:', repr(nc_fid.variables[key].dtype))
            for ncattr in nc_fid.variables[key].ncattrs():
                print ('\t\t%s:' % ncattr,\
                  repr(nc_fid.variables[key].getncattr(ncattr)))
        except KeyError:
            print ('\t\tWARNING: %s does not contain variable attributes' % key)

# NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print ('NetCDF Global Attributes:')
        for nc_attr in nc_attrs:
            print ('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
# Dimension shape information.
    if verb:
        print ('NetCDF dimension information:')
        for dim in nc_dims:
            print ('\tName:', dim)
            print ('\t\tsize:', len(nc_fid.dimensions[dim]))
            print_ncattr(dim)
# Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print ('NetCDF variable information:')
        for var in nc_vars:
            if var not in nc_dims:
                print ('\tName:', var)
                print ('\t\tdimensions:', nc_fid.variables[var].dimensions)
                print ('\t\tsize:', nc_fid.variables[var].size)
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars

nc_f_4 = 'ANO_LONG_CG3D_NA_VAR_zg500_JJA_Y_1000-2000.nc'  # Your filename
nc_fid_4 = Dataset(nc_f_4, 'r')  # Dataset is the class behavior to open the file

nc_attrs_4, nc_dims_4, nc_vars_4 = ncdump(nc_fid_4)

var_zg500 = nc_fid_4.variables['zg'][:]

#ANO_LONG_CG3D_NA_VAR_zg500_JJA_Y_1000-2000.nc contains the variance (scalar- spatial and temporal average) of zg500 in region [80 W- 30 E][30-70N] period 1 June- 30 August for years 1000-2000 data-set with daily cycle

nc_f_5 = 'DATA_FOR_CHAIN_ANO_LONG_CG3D_NA_zg500_JJA_Y_1000-2000.nc'  # Your filename
nc_fid_5 = Dataset(nc_f_5, 'r')  # Dataset is the class behavior to open the file

nc_attrs_5, nc_dims_5, nc_vars_5 = ncdump(nc_fid_5)

lats = nc_fid_5.variables['lat'][:]
lons = nc_fid_5.variables['lon'][:]
zg500_learn = nc_fid_5.variables['zg'][:]

#DATA_FOR_CHAIN_ANO_LONG_CG3D_NA_zg500_JJA_Y_1000-2000.nc contains zg500 in region [80 W- 30 E][30-70N] period 1 June- 27 August for years 1000-2000 data-set with daily cycle (last 3 days excluded)

nc_f_6 = 'ANO_LONG_CG3D_NA_zg500_JJA_Y_1000-2000.nc'  # Your filename
nc_fid_6 = Dataset(nc_f_6, 'r')  # Dataset is the class behavior to open the file

nc_attrs_6, nc_dims_6, nc_vars_6 = ncdump(nc_fid_6)

zg500 = nc_fid_6.variables['zg'][:]

#ANO_LONG_CG3D_NA_zg500_JJA_Y_1000-2000.nc contains zg500 in region [80 W- 30 E][30-70N] period 1 June - 30 August for years 1000-2000 data-set with daily cycle

nc_f_7 = 'gridarea.nc'  # Your filename
nc_fid_7 = Dataset(nc_f_7, 'r')  # Dataset is the class behavior to open the file

nc_attrs_7, nc_dims_7, nc_vars_7 = ncdump(nc_fid_7)

#gridarea.nc contains the area of boxes in region [80 W- 30 E][30-70N] to be used as weights for norm computation

Temp = np.load('France_Anomalies_LONG_JJA_CG3D.npy')

#France_Anomalies_LONG_JJA_CG3D.npy local temperature France

Temp = Temp/np.std(Temp)

Temp = Temp.reshape((len(Temp),1))

Temp_learn = np.zeros((87000,1))

Mrso = np.load('France_Mrso_Anomalies_LONG_JJA_CG3D.npy')

#France_Mrso_Anomalies_LONG_JJA_CG3D.npy local soil moisture France

Mrso = Mrso/np.std(Mrso)

Mrso = Mrso.reshape((len(Mrso),1))

Mrso_learn = np.zeros((87000,1))

#exclude last 3 days of summer from learning data-set
count = 0
for y in range(1000):
    for day in range(y*90,y*90+87):
        Temp_learn[count] = Temp[day]
        Mrso_learn[count] = Mrso[day]
        count += 1

#X_3 and X_3_learn auxiliary vectors for analogues computation

X_3 = np.concatenate((Temp,Mrso),axis=1)

X_3_learn = np.concatenate((Temp_learn,Mrso_learn),axis=1)

print(X_3.shape)

print('Start')

Area = nc_fid_7.variables['cell_area'][:]

Total_Area = Area.sum()

#normalization field with variance and dimension

zg500_norm_learn = zg500_learn / np.sqrt(var_zg500*len(lats)*len(lons))

zg500_norm = zg500 / np.sqrt(var_zg500*len(lats)*len(lons))

print('End Divide, Start creation')

#X and X_learn vectors containing data prepared for analogues computation

X = np.zeros((len(zg500),len(lats)*len(lons)+2)) #2 local variable T and Mrso

X_learn = np.zeros((len(zg500_learn),len(lats)*len(lons)+2)) #2 local variable T and Mrso

#X_2 and X_2_learn auxiliary vectors for analogues computation

X_2 = np.zeros((len(zg500),len(lats)*len(lons)))

X_2_learn = np.zeros((len(zg500_learn),len(lats)*len(lons)))

print('End creation, Start Filling')

#W weights for computing norm
W = np.sqrt(Area/Total_Area)

X_2 = (np.multiply(zg500_norm[:,0,:,:],W)).reshape((len(X_2),-1))

X_2_learn = (np.multiply(zg500_norm_learn[:,0,:,:],W)).reshape((len(X_2_learn),-1))

X = np.concatenate((X_2,X_3),axis=1)
X_learn = np.concatenate((X_2_learn,X_3_learn),axis=1)

print('End Filling')

print('Start Tree creation')

n_neighbors = 1000

tree = cKDTree(X_learn)
dist, ind = tree.query(X, k=n_neighbors,n_jobs = 3)


print('Start print Matrix')

days_learn = 87

days_tot = 90

years = 1000

ind_new = (ind // days_learn)*(days_tot-days_learn) + ind

np.save('Neighbors_Matrix_1000NN.npy',ind_new)

np.save('Distances_Matrix_1000NN.npy',dist)
