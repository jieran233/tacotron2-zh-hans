@echo off
taskkill /F /IM tensorboard.exe
tensorboard --port 6006 --logdir output/log
