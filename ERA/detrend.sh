#This shell script contains a loop that iterates over 
#three variables: zg500, t2m, and water_weighted. 
#For each variable, the script merges multiple files
# with similar names into a single file, selects specific 
#months from the merged file, removes the temporal trend 
#from the selected data, and calculates the anomalies by 
#subtracting the yearly mean from the detrended data. 
#The intermediate files are deleted after each iteration.

for var in zg500 t2m water_weighted; do
    echo cdo -b F32 mergetime Data_ERA5/ERA5_${var}_* Data_ERA5/ERA5_${var}_temp.nc
    cdo -b F32 mergetime Data_ERA5/ERA5_${var}_* Data_ERA5/ERA5_${var}_temp.nc
    echo cdo -selmon,5,6,7,8,9 Data_ERA5/ERA5_${var}_temp.nc Data_ERA5/ERA5_${var}.nc
    cdo -selmon,5,6,7,8,9 Data_ERA5/ERA5_${var}_temp.nc Data_ERA5/ERA5_${var}.nc
    echo rm Data_ERA5/ERA5_${var}_temp.nc 
    rm Data_ERA5/ERA5_${var}_temp.nc 
    echo cdo detrend Data_ERA5/ERA5_${var}.nc ERA5/linear/${var}.nc
    cdo detrend Data_ERA5/ERA5_${var}.nc ERA5/linear/${var}.nc
    echo cdo ydaysub ERA5/linear/${var}.nc -ydaymean ERA5/linear/${var}.nc ERA5/linear/ANO_${var}.nc
    cdo ydaysub ERA5/linear/${var}.nc -ydaymean ERA5/linear/${var}.nc ERA5/linear/ANO_${var}.nc
done