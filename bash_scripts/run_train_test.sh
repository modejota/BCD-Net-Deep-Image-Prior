#!/bin/bash

# Se lanza: nohup ./bash_scripts/run_train_test.sh &

rm -f output/salida_*_train*
rm -f output/salida_*_test*

export CUDA_VISIBLE_DEVICES=0,1

################################################################################################# 

SERVER=dodona
num_workers=32
batch_size=32

#################################################################################################

#python code/train.py --val_type=normal --epochs=1 --cnet_name=unet6sft --batch_size=$batch_size --num_workers=$num_workers --n_samples=10000 > output/salida_${SERVER}_train_${BASHPID}.txt 2>&1
#python code/test.py --cnet_name=unet6sft --batch_size=$batch_size --num_workers=$num_workers --n_samples=10000 > output/salida_${SERVER}_test_${BASHPID}.txt 2>&1

#################################################################################################

mnet_name_array=(mobilenetv3s)
cnet_name_array=(unet6sft unet6)
pretraining_epochs_array=(1)
sigmaRui_sq_array=(0.05)
#theta_val_array=(1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0)
theta_val_array=(0.6 0.5 0.4 0.3 0.2 0.1 0.0)


for cnet_name in "${cnet_name_array[@]}"
do
    for mnet_name in "${mnet_name_array[@]}"
    do
        for pe in "${pretraining_epochs_array[@]}"
        do
            for sigmaRui_sq in "${sigmaRui_sq_array[@]}"
            do
                for theta in "${theta_val_array[@]}"
                do
                    python code/train.py --val_type=GT --sigmaRui_sq=$sigmaRui_sq --batch_size=$batch_size --num_workers=$num_workers --pretraining_epochs=$pe --theta_val=$theta --cnet_name=$cnet_name --mnet_name=$mnet_name > output/salida_${SERVER}_train_${BASHPID}.txt 2>&1
                    #python code/train.py --val_type=normal --sigmaRui_sq=$sigmaRui_sq --batch_size=$batch_size --num_workers=$num_workers --pretraining_epochs=$pe --theta_val=$theta --cnet_name=$cnet_name --mnet_name=$mnet_name > output/salida_${SERVER}_train_${BASHPID}.txt 2>&1
                    python code/test.py --sigmaRui_sq=$sigmaRui_sq --batch_size=$batch_size --num_workers=$num_workers --pretraining_epochs=$pe --theta_val=$theta --cnet_name=$cnet_name --mnet_name=$mnet_name > output/salida_${SERVER}_test_${BASHPID}.txt 2>&1
                done
            done
        done
    done
done