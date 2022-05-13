
for year in  {1979..2020} # {1950..1978} # {1981..2020}
do
    yearlabel=$(printf "%04d" ${year})
    echo python3 ERA5_request_water.py ${yearlabel}
    python3 ERA5_request_water.py ${yearlabel}
done

