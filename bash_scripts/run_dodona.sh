#!/bin/bash

# Se lanza: nohup ./bash_scripts/run_dodona.sh &

rm -f output/salida_dodona_*

export CUDA_VISIBLE_DEVICES=0,1,2

################################################################################################# 

num_workers=32
batch_size=64

pretraining_epochs_array=(0 1)
#lambda_val_array=(0.1 0.25 0.5 0.75 0.9)
lambda_val_array=(0.001 0.01 0.1 1.0)
mnet_name_array=(resnet18in resnet18ft mobilenetv3s mobilenetv3sft)

for pe in "${pretraining_epochs_array[@]}"
do
    for lambda in "${lambda_val_array[@]}"
    do
        for mnet_name in "${mnet_name_array[@]}"
        do
            python code/train.py --batch_size=$batch_size --num_workers=$num_workers --pretraining_epochs=$pe --lambda_val=$lambda --mnet_name=$mnet_name > output/salida_dodona_${BASHPID}.txt 2>&1
            python code/test.py --batch_size=$batch_size --num_workers=$num_workers --pretraining_epochs=$pe --lambda_val=$lambda --mnet_name=$mnet_name > output/salida_dodona_${BASHPID}.txt 2>&1
        done
    done
done