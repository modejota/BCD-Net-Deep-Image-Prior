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

rm -f output/salida_{$SERVER}_train*
rm -f output/salida_{$SERVER}_test*

if [[ "$SERVER" == dodona ]]
then
    echo "Running on dodona"
    export CUDA_VISIBLE_DEVICES=0,1,2
    num_workers=32
    batch_size=48

    #################################################################################################

    #python code/train.py --val_type=normal --epochs=1 --cnet_name=unet6sft --batch_size=$batch_size --num_workers=$num_workers --n_samples=10000 > output/salida_${SERVER}_train_${BASHPID}.txt 2>&1
    #python code/test.py --cnet_name=unet6sft --batch_size=$batch_size --num_workers=$num_workers --n_samples=10000 > output/salida_${SERVER}_test_${BASHPID}.txt 2>&1

    #################################################################################################
    num_epochs=10
    num_runs=1
    mnet_name_array=(mobilenetv3s)
    cnet_name_array=(unet6)
    pretraining_epochs_array=(1)
    #sigmaRui_sq_array=(0.05)
    sigmaRui_sq_array=(0.001 0.05 0.1 0.2)
    #theta_val_array=(1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0)
    theta_val_array=(0.0 0.0 1.0 1.0)

    for ((i=1; i<=$num_runs; i++))
    do
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
                            python code/train.py --epochs=$num_epochs --val_type=GT --sigmaRui_sq=$sigmaRui_sq --batch_size=$batch_size --num_workers=$num_workers --pretraining_epochs=$pe --theta_val=$theta --cnet_name=$cnet_name --mnet_name=$mnet_name > output/salida_${SERVER}_train_${BASHPID}.txt 2>&1
                            #python code/train.py --val_type=normal --sigmaRui_sq=$sigmaRui_sq --batch_size=$batch_size --num_workers=$num_workers --pretraining_epochs=$pe --theta_val=$theta --cnet_name=$cnet_name --mnet_name=$mnet_name > output/salida_${SERVER}_train_${BASHPID}.txt 2>&1
                            python code/test.py --sigmaRui_sq=$sigmaRui_sq --batch_size=$batch_size --num_workers=$num_workers --pretraining_epochs=$pe --theta_val=$theta --cnet_name=$cnet_name --mnet_name=$mnet_name > output/salida_${SERVER}_test_${BASHPID}.txt 2>&1
                        done
                    done
                done
            done
        done
    done
elif [[ "$SERVER" == delfos ]]
then
    echo "Running on delfos"
    export CUDA_VISIBLE_DEVICES=1,2
    num_workers=24
    batch_size=32
    num_epochs=10

    #################################################################################################

    #python code/train.py --val_type=normal --epochs=1 --cnet_name=unet6sft --batch_size=$batch_size --num_workers=$num_workers --n_samples=10000 > output/salida_${SERVER}_train_${BASHPID}.txt 2>&1
    #python code/test.py --cnet_name=unet6sft --batch_size=$batch_size --num_workers=$num_workers --n_samples=10000 > output/salida_${SERVER}_test_${BASHPID}.txt 2>&1

    #################################################################################################

    num_runs=1
    mnet_name_array=(mobilenetv3s)
    cnet_name_array=(unet6)
    pretraining_epochs_array=(1)
    sigmaRui_sq_array=(0.2)
    #theta_val_array=(1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0)
    theta_val_array=(0.3 0.5 0.5 0.7)

    for ((i=1; i<=$num_runs; i++))
    do
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
                            python code/train.py --epochs=$num_epochs --val_type=GT --sigmaRui_sq=$sigmaRui_sq --batch_size=$batch_size --num_workers=$num_workers --pretraining_epochs=$pe --theta_val=$theta --cnet_name=$cnet_name --mnet_name=$mnet_name > output/salida_${SERVER}_train_${BASHPID}.txt 2>&1
                            #python code/train.py --val_type=normal --sigmaRui_sq=$sigmaRui_sq --batch_size=$batch_size --num_workers=$num_workers --pretraining_epochs=$pe --theta_val=$theta --cnet_name=$cnet_name --mnet_name=$mnet_name > output/salida_${SERVER}_train_${BASHPID}.txt 2>&1
                            python code/test.py --sigmaRui_sq=$sigmaRui_sq --batch_size=$batch_size --num_workers=$num_workers --pretraining_epochs=$pe --theta_val=$theta --cnet_name=$cnet_name --mnet_name=$mnet_name > output/salida_${SERVER}_test_${BASHPID}.txt 2>&1
                        done
                    done
                done
            done
        done
    done

    mnet_name_array=(mobilenetv3s)
    cnet_name_array=(unet6)
    pretraining_epochs_array=(1)
    sigmaRui_sq_array=(0.1)
    #theta_val_array=(1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0)
    theta_val_array=(0.5)

    for ((i=1; i<=$num_runs; i++))
    do
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
                            python code/train.py --epochs=$num_epochs --val_type=GT --sigmaRui_sq=$sigmaRui_sq --batch_size=$batch_size --num_workers=$num_workers --pretraining_epochs=$pe --theta_val=$theta --cnet_name=$cnet_name --mnet_name=$mnet_name > output/salida_${SERVER}_train_${BASHPID}.txt 2>&1
                            #python code/train.py --val_type=normal --sigmaRui_sq=$sigmaRui_sq --batch_size=$batch_size --num_workers=$num_workers --pretraining_epochs=$pe --theta_val=$theta --cnet_name=$cnet_name --mnet_name=$mnet_name > output/salida_${SERVER}_train_${BASHPID}.txt 2>&1
                            python code/test.py --sigmaRui_sq=$sigmaRui_sq --batch_size=$batch_size --num_workers=$num_workers --pretraining_epochs=$pe --theta_val=$theta --cnet_name=$cnet_name --mnet_name=$mnet_name > output/salida_${SERVER}_test_${BASHPID}.txt 2>&1
                        done
                    done
                done
            done
        done
    done
fi