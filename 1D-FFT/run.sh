#!/bin/bash
# echo "Argument give is $1"

python preprocessing.py $1

nvcc inv_fft.cu
./a.out $2

python postprocessing.py