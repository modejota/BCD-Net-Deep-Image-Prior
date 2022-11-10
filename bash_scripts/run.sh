#!/bin/bash

# Se lanza: nohup ./bash_scripts/run.sh &

rm -f output/*

export CUDA_VISIBLE_DEVICES=2,3

################################################################################################# 

#python code/train.py --epochs=5 --sigmaRui_sq=0.05 --theta_val=0.5 --pretraining_epochs=0 --n_samples=5000 --save_freq=10 > output/salida_${BASHPID}.txt 2>&1
#python code/test.py --epochs=5 --sigmaRui_sq=0.05 --theta_val=0.5 --pretraining_epochs=0 --n_samples=5000 --save_freq=10 > output/salida_${BASHPID}.txt 2>&1

pretraining_epochs_array=(0 1)
#theta_val_array=(0.0 0.5 0.75 0.9 0.99 0.999 1.0)
theta_val_array=(0.0 0.001 0.01 0.1 0.5 0.9 1.0)


for pe in "${pretraining_epochs_array[@]}"
do
    for theta in "${theta_val_array[@]}"
    do
        python code/train.py --pretraining_epochs=$pe --theta_val=$theta > output/salida_${BASHPID}.txt 2>&1
        python code/test.py --pretraining_epochs=$pe --theta_val=$theta > output/salida_${BASHPID}.txt 2>&1
    done
done