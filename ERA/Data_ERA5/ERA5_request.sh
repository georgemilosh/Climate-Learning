variable='2m_temperature'
variableshort='t2m'

for year in  {1979..2020} #{2020..2020} # {1950..1978} #{1979..2019}
do
    yearlabel=$(printf "%04d" ${year})
    python3 ERA5_request.py ${variable} ${yearlabel}
    cdo daymean download_T2m.nc ERA5_${variableshort}_${yearlabel}.nc
    echo ${year}
done

