#!/bin/bash

# Se lanza: nohup ./bash_scripts/run_deconvolve.sh &

rm -f output/salida_*_deconvolve_*

export CUDA_VISIBLE_DEVICES=0,1,2,3

#################################################################################################

SERVER=delfos
num_workers=32
batch_size=32

#################################################################################################

pretraining_epochs=1
sigmaRui_sq=0.05
theta_val=0.3
mnet_name=mobilenetv3s
dataset_path=/work/Camelyon17/work/DECONVOLUCIONES/Original/
save_path=/work/work_fran/Deep_Var_BCD/results/deconvolutions/Camelyon17/
#dataset_path=/data/BasesDeDatos/Alsubaie/Data/
#save_path=/work/work_fran/Deep_Var_BCD/results/deconvolutions/Alsubaie/

python code/deconvolve.py --sigmaRui_sq=$sigmaRui_sq --batch_size=$batch_size --num_workers=$num_workers --save_path=$save_path --pretraining_epochs=$pretraining_epochs --theta_val=$theta_val --mnet_name=$mnet_name > output/salida_${SERVER}_deconvolve_${BASHPID}.txt 2>&1
