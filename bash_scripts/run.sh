#!/bin/bash

rm -f output/*

export CUDA_VISIBLE_DEVICES=2,3

################################################################################################# 

#exec nohup python code/train.py --epochs=10 --pretraining_epochs=1 --n_samples=1000 --save_freq=2 > output/salida_${BASHPID}.txt &

#exec nohup python code/train.py --pretraining_epochs=5 --lambda_val=1.0 > output/salida_${BASHPID}.txt &

#exec nohup python code/train.py --pretraining_epochs=5 --lambda_val=0.5 > output/salida_${BASHPID}.txt &

exec nohup python code/train.py --pretraining_epochs=5 --lambda_val=0.05 > output/salida_${BASHPID}.txt &