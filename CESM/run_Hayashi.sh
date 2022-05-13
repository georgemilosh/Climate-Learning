#!/bin/bash


fun() {
	var=$1
	set -- $2
	
	for arg1 in $var; do
		echo $arg1 and $1
		for threshold in -100 4.5 3.5; do					
			#echo python3 Hayashi.py $threshold $arg1 $1
			#echo ======================================
			#python3 Hayashi.py $threshold $arg1 $1
			#echo ======================================
			echo python3 plotta_Hayashi.py $threshold $arg1 $1
			echo ======================================
			python3 plotta_Hayashi.py $threshold $arg1 $1
			echo ======================================
		done
		shift
	done
	#set $latssouth # This will ensure that latsouth is also looped through below
	#for latnorth in $latsnorth
	#do
	#	for threshold in -100 4.5 3.5
	#	do
	#		echo python3 Hayashi.py $threshold $1 $latnorth
			#python3 Hayashi.py $threshold $1 $latnorth
	#		echo python3 plotta_Hayashi.py $threshold $1 $latnorth
			#python3 plotta_Hayashi.py $threshold $1 $latnorth
	#		shift
	#	done
	#done
}

fun  "50 60 70 55" "60 70 80 75" #latnorth latsouth
