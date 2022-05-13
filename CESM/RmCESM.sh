#!/bin/bash
# This script removes the concatenatenated files in each folder
FOLDER=Data/
VAR=TSA
PREFIX=lnd.hist.h1
#VAR=Z3
#PREFIX=atm.hist.h2
for batch in $(seq 1 1 10)  # loop accross the batches
do
        if [ $batch -lt 10 ]; then
                NUMFOLDER="000$batch"
        fi
        if [ $batch -eq 10 ]; then
                NUMFOLDER="0010"
        fi
	INPUTFOLDER="CAM4_F2000_p144_ctrl_batch_"$NUMFOLDER.$PREFIX."daily".$VAR
	OUTPUTFILE=$batch.$VAR".nc"
	rm $FOLDER$INPUTFOLDER/$OUTPUTFILE
done
