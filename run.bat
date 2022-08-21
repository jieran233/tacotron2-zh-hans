@ECHO OFF

REM https://stackoverflow.com/a/32395352
set OMP_NUM_THREADS=%NUMBER_OF_PROCESSORS%
set MKL_NUM_THREADS=%NUMBER_OF_PROCESSORS%
set OPENBLAS_NUM_THREADS=%NUMBER_OF_PROCESSORS%

python train.py -c ckpt.pt --output_directory=./output --log_directory=./log
