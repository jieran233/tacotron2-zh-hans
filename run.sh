#!/bin/bash

export MKL_DEBUG_CPU_TYPE=5

siblings=$(grep -c '^processor' /proc/cpuinfo)  # https://stackoverflow.com/a/6481016
export OMP_NUM_THREADS=$siblings
export MKL_NUM_THREADS=$siblings
export OPENBLAS_NUM_THREADS=$siblings
export OMP_NUM_THREADS=$siblings
export MKL_NUM_THREADS=$siblings
export BLIS_NUM_THREADS=$siblings
# export OMP_PROC_BIND=spread
export OPENBLAS_NUM_THREADS=$siblings

python3 train.py -c ckpt.pt --output_directory=./output --log_directory=./log
