#!/bin/bash
# echo "Argument give is $1"

if [ "$#" -ne 3 ]; then
    echo -e "Error : The entered number of arguments is incorrect"
    echo
    echo -e "Usage : ./run_parallel.sh <wav_file> <l|h> <frequency>"
    echo
    echo
    echo -e "\t <wav_file> : This is the file on which the operation is done, in .wav format"
    echo -e "\t <l|h> : l indicates low pass and h indicates high pass"
    echo -e "\t <frequency> : This is the frequency above/below which the signal is cut"

else
    python preprocessing.py $1

    nvcc filter.cu
    ./a.out $2 $3

    python postprocessing.py

fi