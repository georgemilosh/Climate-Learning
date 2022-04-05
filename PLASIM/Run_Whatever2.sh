#!/bin/bash
# This code launches a set of different runs with Learn.py
# It modifies the value in $script
script=Learn2.py
if (grep "tau = 0" $script)
then
	echo "String tau = 0 found"
	for tau in $(seq 0 -5 -30) #-1 -30) #-5 -30)
	do
		echo "tau = "$tau
		sed -i -e 's/tau = 0/tau = '$tau'/g' $script	
		python $script
		sed -i -e 's/tau = '$tau'/tau = 0/g' $script	
	done
else
	echo "String tau = 0 not found!"
fi
