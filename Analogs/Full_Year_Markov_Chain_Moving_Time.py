import datetime as dt  # Python standard library datetime  module
import numpy as np
from netCDF4 import Dataset
from scipy.spatial import cKDTree
#import cdo

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

nc_f_4 = 'DAILY_CG_5DAYS_NORTH_ATLANTIC_GLOBAL_VAR_zg500.nc'  # Your filename
nc_fid_4 = Dataset(nc_f_4, 'r')  # Dataset is the class behavior to open the file

nc_attrs_4, nc_dims_4, nc_vars_4 = ncdump(nc_fid_4)

#DAILY_CG_5DAYS_NORTH_ATLANTIC_GLOBAL_VAR_zg500.nc contains the variance (scalar- spatial and temporal average) of zg500 in region [80 W- 30 E][30-70N] period 3 January year 0 - 28 December year 99 (centered coarse-graining)  data-set without daily cycle


var_zg500 = nc_fid_4.variables['zg'][:]

nc_f_6 = 'DAILY_CG_5DAYS_NORTH_ATLANTIC_zg500.nc'  # Your filename
nc_fid_6 = Dataset(nc_f_6, 'r')  # Dataset is the class behavior to open the file

nc_attrs_6, nc_dims_6, nc_vars_6 = ncdump(nc_fid_6)

lats = nc_fid_6.variables['lat'][:]
lons = nc_fid_6.variables['lon'][:]
zg500 = nc_fid_6.variables['zg'][:]

#DAILY_CG_5DAYS_NORTH_ATLANTIC_zg500.nc contains zg500 in region [80 W- 30 E][30-70N] period 3 January year 0 - 28 December year 99 (centered coarse-graining)  data-set without daily cycle

nc_f_7 = 'gridarea.nc'  # Your filename
nc_fid_7 = Dataset(nc_f_7, 'r')  # Dataset is the class behavior to open the file

nc_attrs_7, nc_dims_7, nc_vars_7 = ncdump(nc_fid_7)

Area = nc_fid_7.variables['cell_area'][:]

Total_Area = Area.sum()

W = np.sqrt(Area / Total_Area)

#gridarea.nc contains the area of boxes in region [80 W- 30 E][30-70N] to be used as weights for norm computation

#normalization field with variance
zg500_norm = zg500 / np.sqrt(var_zg500)

X = np.zeros((len(zg500),len(lats)*len(lons)))

num_century = 10

num_days_per_century = 360*100-4

num_month = 12

num_days_per_month = 30

day_window = 120

num_days_per_year = num_month*num_days_per_month

#X_learn = np.zeros((num_years*(day_window + 1),len(lats)*len(lons)))

for n in range(len(zg500)): #Convert Zg to 2 Dim array
    X[n] = np.reshape(np.multiply(zg500_norm[n][0],W),len(lats)*len(lons))

n_neighbors = 2000

T = np.zeros((len(X),n_neighbors),dtype = 'int')

Neighbors_Matrix = open('Neighbors_Matrix_2000NN_Moving_Time_Seasonal.dat','w')

#Preparation Data for analogues computation
#Cycles running on months and days: several cases since there are missing dates some years (for instance 1 and 2 January or 29 and 30 December). Furthermore the moving time window could be smaller than 60 days for years at start/end of each batch.

for month in range(num_month):
#for month in range(1):
#    for j in range(num_days_per_century):
    for day in range(num_days_per_month):
        X_New = []
        X_learn = []
        pos = []
        pos_new = []
        if(month!=9 and month!=10 and month!=11 and month!=2 and month!=1 and month!=0):#60 days moving average without exception
            for cent in range(num_century):
                for y in range(100):
                    X_New.append(X[(cent*100+y)*num_days_per_year+month*num_days_per_month+day-2-4*cent])
                    pos_new.append((cent*100+y)*num_days_per_year+month*num_days_per_month+day-2-4*cent)
                    for l in range(day_window+1):
                        X_learn.append(X[(cent*100+y)*num_days_per_year+month*num_days_per_month+(day-60+l)-2-4*cent])
                        pos.append((cent*100+y)*num_days_per_year+month*num_days_per_month+(day-60+l)-2-4*cent)
        #Start exceptions
        elif(month == 2):
            for cent in range(num_century):
                for y in range(100):
                    X_New.append(X[(cent*100+y)*num_days_per_year+month*num_days_per_month+day-2-4*cent])
                    pos_new.append((cent*100+y)*num_days_per_year+month*num_days_per_month+day-2-4*cent)
                    if(y == 0):
                        start = max(0,2-day)
                    else:
                        start = 0
                    for l in range(start,day_window+1):
                        X_learn.append(X[(cent*100+y)*num_days_per_year+month*num_days_per_month+(day-60+l)-2-4*cent])
                        pos.append((cent*100+y)*num_days_per_year+month*num_days_per_month+(day-60+l)-2-4*cent)
        elif(month == 9):
            for cent in range(num_century):
                for y in range(100):
                    X_New.append(X[(cent*100+y)*num_days_per_year+month*num_days_per_month+day-2-4*cent])
                    pos_new.append((cent*100+y)*num_days_per_year+month*num_days_per_month+day-2-4*cent)
                    if(y == 99):
                        end = max(0,day-22)
                    else:
                        end = 0
                    for l in range(day_window+1-end):
                        X_learn.append(X[(cent*100+y)*num_days_per_year+month*num_days_per_month+(day-60+l)-2-4*cent])
                        pos.append((cent*100+y)*num_days_per_year+month*num_days_per_month+(day-60+l)-2-4*cent)
        elif(month == 0 or month==1):
            for cent in range(num_century):
                for y in range(100):
                    if(y == 0):
                        start = 62-30*month-day
                        if(day>1):
                            X_New.append(X[(cent*100+y)*num_days_per_year+month*num_days_per_month+day-2-4*cent])
                            pos_new.append((cent*100+y)*num_days_per_year+month*num_days_per_month+day-2-4*cent)
                        elif(month==1):
                            X_New.append(X[(cent*100+y)*num_days_per_year+month*num_days_per_month+day-2-4*cent])
                            pos_new.append((cent*100+y)*num_days_per_year+month*num_days_per_month+day-2-4*cent)
                    else:
                        start = 0
                        X_New.append(X[(cent*100+y)*num_days_per_year+month*num_days_per_month+day-2-4*cent])
                        pos_new.append((cent*100+y)*num_days_per_year+month*num_days_per_month+day-2-4*cent)
                    for l in range(start,day_window+1):
                        X_learn.append(X[(cent*100+y)*num_days_per_year+month*num_days_per_month+(day-60+l)-2-4*cent])
                        pos.append((cent*100+y)*num_days_per_year+month*num_days_per_month+(day-60+l)-2-4*cent)
        elif(month == 11 or month == 10):
            for cent in range(num_century):
                for y in range(100):
                    if(y == 99):
                        end = day_window+day+30*(month-9)-112
                        if(day<28):
                            X_New.append(X[(cent*100+y)*num_days_per_year+month*num_days_per_month+day-2-4*cent])
                            pos_new.append((cent*100+y)*num_days_per_year+month*num_days_per_month+day-2-4*cent)
                        elif(month==10):
                            X_New.append(X[(cent*100+y)*num_days_per_year+month*num_days_per_month+day-2-4*cent])
                            pos_new.append((cent*100+y)*num_days_per_year+month*num_days_per_month+day-2-4*cent)
                    else:
                        end = 0
                        X_New.append(X[(cent*100+y)*num_days_per_year+month*num_days_per_month+day-2-4*cent])
                        pos_new.append((cent*100+y)*num_days_per_year+month*num_days_per_month+day-2-4*cent)
                    for l in range(day_window+1-end):
                        X_learn.append(X[(cent*100+y)*num_days_per_year+month*num_days_per_month+(day-60+l)-2-4*cent])
                        pos.append((cent*100+y)*num_days_per_year+month*num_days_per_month+(day-60+l)-2-4*cent)
        print(len(X_learn))
        X_learn = np.asarray(X_learn)
        X_New = np.asarray(X_New)
        
        tree = cKDTree(X_learn) #Tree creation
        
        #Analogues computation
        dist, ind = tree.query(X_New, k=n_neighbors)
        for l in range(len(X_New)):
            for m in range(n_neighbors):
                T[pos_new[l]][m] = pos[ind[l][m]]
                #D[pos_new[l]][m] = dist[l][m]
                #Neighbors_Matrix.write('%d\t'%pos[ind[l][m]])
            #Neighbors_Matrix.write('\n')

#print Matrix analogues
for i in range(len(X)):
    for j in range(n_neighbors):
        Neighbors_Matrix.write('%d\t'%T[i][j])
        #Distances_Matrix.write('%lf\t'%(D[i][j]))
    Neighbors_Matrix.write('\n')
    #Distances_Matrix.write('\n')
        
