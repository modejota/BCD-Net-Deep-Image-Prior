#!/bin/bash

# Se lanza: nohup ./bash_scripts/run_dodona.sh &

rm -f output/salida_dodona_*

export CUDA_VISIBLE_DEVICES=0,1,2

################################################################################################# 

num_workers=32
batch_size=64

mnet_name_array=(mobilenetv3s)
pretraining_epochs_array=(0)
sigmaRui_sq_array=(0.05)
theta_val_array=(1.0 0.9 0.8 0.6 0.4 0.3 0.1 0.0)

for mnet_name in "${mnet_name_array[@]}"
do
    for pe in "${pretraining_epochs_array[@]}"
    do
        for sigmaRui_sq in "${sigmaRui_sq_array[@]}"
        do
            for theta in "${theta_val_array[@]}"
            do
                python code/train.py --val_type=GT --sigmaRui_sq=$sigmaRui_sq --batch_size=$batch_size --num_workers=$num_workers --pretraining_epochs=$pe --theta_val=$theta --mnet_name=$mnet_name > output/salida_dodona_${BASHPID}.txt 2>&1
                python code/test.py --sigmaRui_sq=$sigmaRui_sq --batch_size=$batch_size --num_workers=$num_workers --pretraining_epochs=$pe --theta_val=$theta --mnet_name=$mnet_name > output/salida_dodona_${BASHPID}.txt 2>&1
            done
        done
    done
done

mnet_name_array=(resnet18ft)
pretraining_epochs_array=(0)
sigmaRui_sq_array=(0.05)
theta_val_array=(1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0)

for mnet_name in "${mnet_name_array[@]}"
do
    for pe in "${pretraining_epochs_array[@]}"
    do
        for sigmaRui_sq in "${sigmaRui_sq_array[@]}"
        do
            for theta in "${theta_val_array[@]}"
            do
                python code/train.py --val_type=GT --sigmaRui_sq=$sigmaRui_sq --batch_size=$batch_size --num_workers=$num_workers --pretraining_epochs=$pe --theta_val=$theta --mnet_name=$mnet_name > output/salida_dodona_${BASHPID}.txt 2>&1
                python code/test.py --sigmaRui_sq=$sigmaRui_sq --batch_size=$batch_size --num_workers=$num_workers --pretraining_epochs=$pe --theta_val=$theta --mnet_name=$mnet_name > output/salida_dodona_${BASHPID}.txt 2>&1
            done
        done
    done
done