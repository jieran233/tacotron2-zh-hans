#!/bin/bash

siblings=$(grep -c '^processor' /proc/cpuinfo)  # https://stackoverflow.com/a/6481016
export OMP_NUM_THREADS=$siblings
export MKL_NUM_THREADS=$siblings
export OPENBLAS_NUM_THREADS=$siblings
python3 train.py -c ckpt.pt --output_directory=./output/ckpt --log_directory=./output/log
