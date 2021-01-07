#!/bin/bash
# $1 GPU Device ID
# $2 Model Type (DeepXML/DeepXML-OVA etc.)
# $3 Dataset
# $4 version
# $5 seed
# eg. ./run_main.sh 0 DeepXML EURLex-4K 0 22
# eg. ./run_main.sh 0 DeepXML-fr EURLex-4K 0 22

export CUDA_VISIBLE_DEVICES=$1
model_type=$2
dataset=$3
version=$4
seed=$5

work_dir=$(cd ../../../../ && pwd)

current_working_dir=$(pwd)
python3 ../runner.py "${model_type}" "${work_dir}" ${version} "$(dirname "$current_working_dir")/configs/${model_type}/${dataset}.json" "${seed}"
