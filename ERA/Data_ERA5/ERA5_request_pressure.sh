variable='geopotential'
variableshort='zg500'

for year in {1979..2020}  #{2020..2020} # {1950..1978} # {1979..2019}
do
    echo ${year}
    yearlabel=$(printf "%04d" ${year})
    python3 ERA5_request_pressure.py ${variable} ${yearlabel}
    cdo daymean download_zg500.nc ERA5_${variableshort}_${yearlabel}.nc
done

