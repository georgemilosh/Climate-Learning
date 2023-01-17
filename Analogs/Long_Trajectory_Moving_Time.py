import datetime as dt  # Python standard library datetime  module
import numpy as np
from scipy.stats import norm
from scipy.stats import kurtosis
from scipy.stats import skew
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from sklearn.neighbors import KDTree
import cdo
import random

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
    
#load matrix
T = np.loadtxt('Neighbors_Matrix_2000NN_Moving_Time_Seasonal.dat',dtype = 'int')

n_neighbors = 5

print(len(T))

states = [] #placeholder for distribution of occurence

num_days_per_century = 360*100-4

num_days_per_year = 360

Traj = open('TrajectoryNN%s.dat'%(str(n_neighbors)),'w')

#If one wants to change trajectory's length or try differents lengths
for j in range(5,6):
    
    N = 10**(j+1) #total step Markov chain
    cent = random.randint(0,9)
    year = random.randint(1,99)
    #s=cent*num_days_per_century+year*num_days_per_year-2-4*cent
    s = 358 #initial state 1 January year 1
    #day = cent*num_days_per_century+year*num_days_per_year-((cent*num_days_per_century+year*num_days_per_year)//num_days_per_year)*num_days_per_year
    day = 0
    time = 0
    states.append(day)
    Traj.write('%d\t%d\t%d\n'%(time,day,s))
    #start dynamics
    for i in range(N):
        count = 0
        num_neigh_selected = 0
        moving_neighbors = []
        
        while(num_neigh_selected<n_neighbors and count<2000):
            cent = T[s][count]//num_days_per_century
            day_n = T[s][count]+2+4*cent-((T[s][count]+2+4*cent)//num_days_per_year)*num_days_per_year
            if(time>=30 and time<330):
                if(day_n>=time-30 and day_n<=time+30):
                    num_neigh_selected += 1
                    moving_neighbors.append(T[s][count])
            else:
                if(time<30):
                    if(day_n>=num_days_per_year+time-30 and day_n<num_days_per_year):
                        num_neigh_selected += 1
                        moving_neighbors.append(T[s][count])
                    elif(day_n>=0 and day_n<=time + 30):
                        num_neigh_selected += 1
                        moving_neighbors.append(T[s][count])
                else:
                    if(day_n>=0 and day_n<=time+30-num_days_per_year):
                        num_neigh_selected += 1
                        moving_neighbors.append(T[s][count])
                    elif(day_n>=time-30 and day_n<num_days_per_year):
                        num_neigh_selected += 1
                        moving_neighbors.append(T[s][count])
            count += 1
            
        print(len(moving_neighbors))
        if(len(moving_neighbors)<n_neighbors):
            print(time,day)
        app = random.randint(0,n_neighbors-1)
        s = moving_neighbors[app]+5
        time += 5
        if(time>=360):
            time = time - 360
        cent = s//num_days_per_century
        day = s+2+4*cent-((s+2+4*cent)//num_days_per_year)*num_days_per_year
        states.append(day)
        Traj.write('%d\t%d\t%d\n'%(time,day,s))
    fig = plt.figure()

    plt.hist(states,bins=range(0,361),density=True)
    plt.xlabel('day')
    plt.ylabel('Distribution of days')
    plt.title('Occurence of each day in a long trajectory')
    fig.savefig('Distribution_days_%s_%s.png'%(str(j),str(n_neighbors)))
    plt.close(fig)

