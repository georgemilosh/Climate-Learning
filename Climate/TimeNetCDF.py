# read the time length of .nc files
from netCDF4 import Dataset
import sys
import numpy as np

import glob, os

os.system("export OMP_NUM_THREADS=1")

os.chdir(sys.argv[1])
i = 0
totlength = 0
for file in glob.glob("*.nc"):
    i+=1    
    nc_fid = Dataset(sys.argv[1]+"/"+file, 'r')  # Dataset is the class behavior to open the file
    length = len(nc_fid.variables['time'][:])
    totlength += length
    if length < 365:
        print("insufficient in " + file+": "+str(length))
    nc_fid.close()
print("Found "+str(i)+" .nc files with the total length "+str(totlength))
