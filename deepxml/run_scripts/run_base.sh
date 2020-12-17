#!/bin/bash

train () {
        # $1 model_dir
        # $2 result_dir
        # $3 Extra Parameters
        
        log_tr_file="${2}/log_train.txt"
        python -u ${work_dir}/programs/deepxml/deepxml/main.py --model_dir $1 \
                                --result_dir $2 \
                                --mode train \
                                ${3}| tee $log_tr_file
}


predict () {
        # $1 result_dir
        # $2 model_dir
        # $3 extra params
        log_pred_file="${1}/log_predict.txt"
        python -u ${work_dir}/programs/deepxml/deepxml/main.py --model_dir $2 \
                                --result_dir $1 \
                                --mode predict \
                                ${3} | tee $log_pred_file
}


extract () {
        # $1 result_dir
        # $2 model_dir
        # $3 extra params
        log_pred_file="${1}/log_predict.txt"
        python -u ${work_dir}/programs/deepxml/deepxml/main.py --result_dir $1 \
                                --model_dir $2 \
                                --mode extract \
                                ${3} | tee -a $log_pred_file
}


evaluate () {
        # $1 result_dir
        # $2 train
        # $3 test
        # $4 pred_fname path
        # $5 A
        # $6 B
        # $7 SAVE and BETAS
        log_eval_file="${1}/log_eval.txt"
        python -u ${work_dir}/programs/deepxml/deepxml/tools/evaluate.py "${2}" "${3}" "${4}" $5 $6 $7 | tee -a $log_eval_file
}


# $1 Flag
# $2 dataset
# $3 work directory
# $4 version

FLAG="${1}"
dataset=$2
work_dir=$3
version=$4
model_name=$5
data_dir="${3}/data/${2}"
model_dir="${3}/models/${model_name}/${2}/v_${4}"
result_dir="${3}/results/${model_name}/${2}/v_${4}"
shift 5

mkdir -p $model_dir
mkdir -p $result_dir

if [ "${FLAG}" == "train" ]
then
    # $1 PARAMS
    train $model_dir $result_dir "${1}"
elif [ "${FLAG}" == "predict" ]
then
    # #1 PARAMS
    predict $result_dir $model_dir "${1}"
elif [ "${FLAG}" == "evaluate" ]
then
    # $1 Out_file
    # $2 A
    # $3 B
    # $4 TYPE, SAVE and BETAS
    evaluate $result_dir "${data_dir}/trn_X_Y.txt" "${data_dir}/tst_X_Y.txt" "${result_dir}/${1}" ${2} ${3} "${4}"
elif [ "${FLAG}" == "extract" ]
then
    # $1 PARAMS
    mkdir -p "${result_dir}/export"
    extract $result_dir $model_dir "${1}"
else
    echo "Unknown flag in run_base"
    exit
fi
