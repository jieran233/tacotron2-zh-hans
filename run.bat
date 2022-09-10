@ECHO OFF

SET MKL_DEBUG_CPU_TYPE=5

REM https://stackoverflow.com/a/32395352
SET OMP_NUM_THREADS=%NUMBER_OF_PROCESSORS%
SET MKL_NUM_THREADS=%NUMBER_OF_PROCESSORS%
SET BLIS_NUM_THREADS=%NUMBER_OF_PROCESSORS%
REM SET OMP_PROC_BIND=spread
SET OPENBLAS_NUM_THREADS=%NUMBER_OF_PROCESSORS%


python train.py -c ckpt.pt --output_directory=./output --log_directory=./log
