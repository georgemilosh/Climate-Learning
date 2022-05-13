
#!/bin/bash
BATCH="0010"
FOLDER=Data_CESM/CAM4_F2000_p144_ctrl_batch_$BATCH.daily.Z3.500hPa/
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

	echo mv $FOLDER"CAM4_F2000_p144_ctrl_batch_"$BATCH.Z3.500hPa.daily.$NUMFILE.nc  $FOLDER"CAM4_F2000_p144_ctrl_batch_"$BATCH.daily.Z3.500hPa.$NUMFILE.nc
	mv $FOLDER"CAM4_F2000_p144_ctrl_batch_"$BATCH.Z3.500hPa.daily.$NUMFILE.nc  $FOLDER"CAM4_F2000_p144_ctrl_batch_"$BATCH.daily.Z3.500hPa.$NUMFILE.nc
done
