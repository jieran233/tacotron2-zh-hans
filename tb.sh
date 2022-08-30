#!/bin/bash

killall tensorboard
tensorboard --port 6006 --logdir output/log
