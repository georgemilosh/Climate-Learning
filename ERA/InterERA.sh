#!/bin/bash
# This script goes inside $1 folder provided by the argument when it is called
# It extracts only mrso variable and puts it inside /scratch/gmiloshe/test/ directory
SCRATCH=/scratch/gmiloshe/CDS/Data_ERA5/
GEORGE=$SCRATCH
#VAR=t2m
#VAR=zg500
VAR=water_weighted
for years in $(seq 1950 1 2020)  #$(seq 1950 1 1978)  # $(seq 1979 1 2019)
do
	INPUTFILE="ERA5_"$VAR"_"$years".nc"
	OUTPUTFILE="inter_"$INPUTFILE
	echo cdo  -runmean,3 $GEORGE$INPUTFILE $GEORGE$OUTPUTFILE 
	cdo  -runmean,3 $GEORGE$INPUTFILE $GEORGE$OUTPUTFILE 
done

