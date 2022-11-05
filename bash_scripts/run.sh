#!/bin/bash

# Se lanza: nohup ./bash_scripts/run.sh &

rm -f output/*

export CUDA_VISIBLE_DEVICES=1

################################################################################################# 

python code/train.py --epochs=5 --pretraining_epochs=3 --n_samples=1000 --save_freq=2 > output/salida_${BASHPID}.txt 2>&1

#python code/train.py --pretraining_epochs=5 --sigmaRui_sq=0.001 --lambda_val=1.0 > output/salida_${BASHPID}.txt 2>&1
#python code/test.py --pretraining_epochs=5 --sigmaRui_sq=0.001 --lambda_val=1.0 > output/salida_${BASHPID}.txt 2>&1

#python code/train.py --pretraining_epochs=5 --sigmaRui_sq=0.001 --lambda_val=0.05 > output/salida_${BASHPID}.txt 2>&1
#python code/test.py --pretraining_epochs=5 --sigmaRui_sq=0.001 --lambda_val=0.05 > output/salida_${BASHPID}.txt 2>&1

#python code/train.py --pretraining_epochs=1 --sigmaRui_sq=0.05 --lambda_val=0.005 > output/salida_${BASHPID}.txt 2>&1
#python code/test.py --pretraining_epochs=1 --sigmaRui_sq=0.05 --lambda_val=0.005 > output/salida_${BASHPID}.txt 2>&1

#python code/train.py --pretraining_epochs=0 --sigmaRui_sq=0.001 --lambda_val=0.005 > output/salida_${BASHPID}.txt 2>&1
#python code/test.py --pretraining_epochs=0 --sigmaRui_sq=0.001 --lambda_val=0.005 > output/salida_${BASHPID}.txt 2>&1