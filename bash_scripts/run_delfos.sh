#!/bin/bash

# Se lanza: nohup ./bash_scripts/run_delfos.sh &

rm -f output/salida_delfos_*

export CUDA_VISIBLE_DEVICES=0,1,2

#################################################################################################

num_workers=16
batch_size=32

#################################################################################################

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
                python code/train.py --val_type=GT --sigmaRui_sq=$sigmaRui_sq --batch_size=$batch_size --num_workers=$num_workers --pretraining_epochs=$pe --theta_val=$theta --mnet_name=$mnet_name > output/salida_delfos_${BASHPID}.txt 2>&1
                python code/test.py --sigmaRui_sq=$sigmaRui_sq --batch_size=$batch_size --num_workers=$num_workers --pretraining_epochs=$pe --theta_val=$theta --mnet_name=$mnet_name > output/salida_delfos_${BASHPID}.txt 2>&1
            done
        done
    done
done

#################################################################################################

<<comment
pretraining_epochs=0
theta_val=0.1
dataset_path=/work/Camelyon17/work/DECONVOLUCIONES/Original/
save_path=/work/work_fran/Deep_Var_BCD/results/deconvolutions/Camelyon17/
#dataset_path=/data/BasesDeDatos/Alsubaie/Data/
#save_path=/work/work_fran/Deep_Var_BCD/results/deconvolutions/Alsubaie/

python code/deconvolve.py --batch_size=$batch_size --num_workers=$num_workers --save_path=$save_path --pretraining_epochs=$pretraining_epochs --theta_val=$theta_val > output/salida_delfos_${BASHPID}.txt 2>&1
comment