#!/bin/bash

# Se lanza: nohup ./bash_scripts/run_train_test.sh delfos/dodona &

################################################################################################# 

export CUDA_DEVICE_ORDER=PCI_BUS_ID

SERVER=$1

if [[ "$SERVER" == "" ]]
then
    echo "ERROR: No se ha especificado el servidor"
    exit
fi

if [[ "$SERVER" == dodona ]]
then
    echo "Running on dodona"
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    num_workers=24
    batch_size=128

    num_epochs=50
    num_runs=1
    theta_val_array=(0.3)

    for ((i=1; i<=$num_runs; i++))
    do
        for theta in "${theta_val_array[@]}"
        do
            python code/main.py --use_wandb --mode=train_test --epochs=$num_epochs --batch_size=$batch_size --num_workers=$num_workers --theta_val=$theta --pretrain_epochs=3 --pretrain_theta_val=1.0 --lr=0.0001 > output/salida_${SERVER}_train_${BASHPID}.txt 2>&1
        done
    done
elif [[ "$SERVER" == delfos ]]
then
    echo "Running on delfos"
    export CUDA_VISIBLE_DEVICES=0,1
    num_workers=24
    batch_size=64

    num_epochs=25
    num_runs=1
    theta_val_array=(0.6 0.7 0.8 0.9 1.0)

    for ((i=1; i<=$num_runs; i++))
    do
        for theta in "${theta_val_array[@]}"
        do
            python code/main.py --use_wandb --mode=train_test --epochs=$num_epochs --batch_size=$batch_size --num_workers=$num_workers --theta_val=$theta > output/salida_${SERVER}_train_${BASHPID}.txt 2>&1
        done
    done
fi