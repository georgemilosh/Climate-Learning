#!/bin/bash
# This script runs classification for certain checkpoints
#folder=./models/maskFrance/Z64/yrs100/k.9.1fw2.1.2.lrs8skip2/
folder=$1
for checkpoint in $(seq 2000 100 9900)
#for checkpoint in $(seq 0 10 500)
#for checkpoint in $(seq 500 100 1000)
do
    if [ $checkpoint -eq 0 ]; then
		checkpoint=1
	fi
	#echo "checkpoint = "$checkpoint
    echo python classification.py $folder $checkpoint
    python classification.py $folder $checkpoint
done
