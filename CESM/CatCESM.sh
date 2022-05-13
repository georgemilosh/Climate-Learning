#!/bin/bash
# This script concatenates the files in each folder along the years
# Then it selects the North hemisphere section
FOLDER=Data_CESM/
#VAR=TSA
#PREFIX=.lnd.hist.h1
#VAR=Z3.500hPa
#PREFIX=""
VAR=H2OSOI
PREFIX=.lnd.hist.h0
export OMP_NUM_THREADS=1 # make sure we are only running on one core

COLLECT="" 
for batch in $(seq 1 1 10)  # loop accross the batches
do
        if [ $batch -lt 10 ]; then
                NUMFOLDER="000$batch"
        fi
        if [ $batch -eq 10 ]; then
                NUMFOLDER="0010"
        fi
	INPUTFOLDER="CAM4_F2000_p144_ctrl_batch_"$NUMFOLDER$PREFIX."daily".$VAR
        COMMAND=""	
	for year in $(seq 1 1 100)
	do
		if [ $year -lt 10 ]; then
			NUMFILE="000$year"
                else
			if [ $year -lt 100 ]; then
				NUMFILE="00$year"
			else
				NUMFILE="0$year"
			fi
		fi

		INPUTFILE=$INPUTFOLDER.$NUMFILE".nc"
		COMMAND=$COMMAND$FOLDER$INPUTFOLDER/$INPUTFILE" "
	done
	OUTPUTFILE=$batch.$VAR".nc"
	echo cdo -O mergetime $COMMAND $FOLDER$INPUTFOLDER/"temp"$OUTPUTFILE # so that we can select months later
	cdo -O mergetime $COMMAND $FOLDER$INPUTFOLDER/"temp"$OUTPUTFILE # so that we can select months later
	#echo  cdo -selmon,5,6,7,8,9 $FOLDER$INPUTFOLDER/"temp"$OUTPUTFILE $FOLDER$INPUTFOLDER/$OUTPUTFILE
	#cdo -selmon,5,6,7,8,9 $FOLDER$INPUTFOLDER/"temp"$OUTPUTFILE $FOLDER$INPUTFOLDER/$OUTPUTFILE
	echo  cdo -selmon,5,6,7,8,9 $FOLDER$INPUTFOLDER/"temp"$OUTPUTFILE $FOLDER$INPUTFOLDER/$OUTPUTFILE #"temp2"OUTPUTFILE
	cdo -selmon,5,6,7,8,9 $FOLDER$INPUTFOLDER/"temp"$OUTPUTFILE $FOLDER$INPUTFOLDER/$OUTPUTFILE  #"temp2"$OUTPUTFILE
	echo rm $FOLDER$INPUTFOLDER/"temp"$OUTPUTFILE
	rm $FOLDER$INPUTFOLDER/"temp"$OUTPUTFILE
	#echo cdo sellonlatbox,180,-180,90,30 $FOLDER$INPUTFOLDER/"temp2"$OUTPUTFILE $FOLDER$INPUTFOLDER/$OUTPUTFILE
	#cdo sellonlatbox,180,-180,90,30 $FOLDER$INPUTFOLDER/"temp2"$OUTPUTFILE $FOLDER$INPUTFOLDER/$OUTPUTFILE
	#echo rm $FOLDER$INPUTFOLDER/"temp2"$OUTPUTFILE
	#rm $FOLDER$INPUTFOLDER/"temp2"$OUTPUTFILE
	COLLECT=$COLLECT$FOLDER$INPUTFOLDER/$OUTPUTFILE" "
done
echo rm $FOLDER"North_"$VAR".nc"
rm $FOLDER"North_"$VAR".nc"
echo cdo cat $COLLECT $FOLDER"North_"$VAR".nc"
cdo cat $COLLECT $FOLDER"North_"$VAR".nc"

echo rm $COLLECT # remove the extra stuff
rm $COLLECT # remove the extra stuff
echo rm $FOLDER"North_Anomalies_"$VAR".nc"
rm $FOLDER"North_Anomalies_"$VAR".nc"
echo cdo ydaysub $FOLDER"North_"$VAR".nc" -ydaymean $FOLDER"North_"$VAR".nc" $FOLDER"North_Anomalies_"$VAR".nc"
cdo ydaysub $FOLDER"North_"$VAR".nc" -ydaymean $FOLDER"North_"$VAR".nc" $FOLDER"North_Anomalies_"$VAR".nc"
