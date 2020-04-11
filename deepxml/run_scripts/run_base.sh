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

retrain () {
        # $1 dataset
        # $2 data_dir
        # $3 model_dir
        # $4 result_dir
        # $5 model_fname
        # $6 batch_size
        # $7 pred_fname
        # $14 extra params

        log_pred_file="${4}/log_predict.txt"
        python -u ${work_dir}/programs/deepxml/deepxml/main.py --dataset $1 \
                                --data_dir $2 \
                                --model_dir $3 \
                                --result_dir $4 \
                                --embedding_dims $6 \
                                --batch_size ${12} \
                                --lr $5 \
                                --dlr_step ${11} \
                                --model_fname ${13} \
                                --dlr_factor ${10} \
                                --vocabulary_dims $7 \
                                --num_labels $8 \
                                --num_epochs $9\
                                ${14} |& tee $log_tr_file
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


retrain_w_shorty () {
        # $1 model_dir
        # $2 result_dir
        # $3 Extra Parameters
        
        log_tr_file="${2}/log_train_post.txt"
        python -u ${work_dir}/programs/deepxml/deepxml/main.py --model_dir $1 \
                                --result_dir $2 \
                                --mode retrain_w_shorty \
                                ${3}| tee $log_tr_file
}

evaluate () {
        # $1 result_dir
        # $2 train
        # $3 test
        # $4 pred_fname path
        # $5 A
        # $6 B
        # $7 TYPE SAVE and BETAS
        log_eval_file="${1}/log_eval.txt"
        python -u ${work_dir}/programs/deepxml/deepxml/tools/evaluate.py "${2}" "${3}" "${4}" $5 $6 $7 | tee -a $log_eval_file
}

post_process(){
    # $1 data directory
    # $2 embedding files
    # $3 file name
    # $4 result directory
    data_dir=$1
    wrd_emb=$2
    file=$3
    result_dir=$4
    python -u ${work_dir}/programs/deepxml/deepxml/tools/gen_dense_dataset.py "${data_dir}/${file}.txt" "${wrd_emb}" "${result_dir}/${file}.txt"
    wait
}

gen_tail_emb ()
{
    # $1 data directory
    # $2 result directory
    # $3 version
    # $4 embedding_dims

    data_dir=$1
    result_dir=$2
    model_dir="$(dirname "$3")"
    version=$4
    embedding_dims=$5
    temp_dir=$6
    
    original_emb="${data_dir}/fasttextB_embeddings_${embedding_dims}d.npy"
    gen_emb=$result_dir
    feat_idx="${data_dir}/${temp_dir}/features_split_${version}.txt"
    out_emb="${model_dir}/head_embeddings_${embedding_dims}d.npy"
    python ${work_dir}/programs/deepxml/deepxml/tools/init_embedding_from_head.py $original_emb $gen_emb $feat_idx $out_emb
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
    evaluate $result_dir $data_dir'/trn_X_Y.txt' $data_dir'/tst_X_Y.txt' "${result_dir}/${1}" ${2} ${3} "${4}"

elif [ "${FLAG}" == "extract" ]
then
    # $1 PARAMS
    mkdir -p "${result_dir}/export"
    extract $result_dir $model_dir "${1}"


elif [ "${FLAG}" == "postprocess" ]
then
    # $1 embedding files
    # $2 file 
    post_process $data_dir "${result_dir}/${1}" "${2}" $result_dir/export


elif [ "${FLAG}" == "retrain_w_shortlist" ]
then
    # $1 embedding files
    # $2 file 
    retrain_w_shorty $model_dir $result_dir "${1}"


elif [ "${FLAG}" == "gen_tail_emb" ]
then
    # $1 embedding files
    # $2 file 
    gen_tail_emb $data_dir $result_dir/$1 $model_dir $2 $3 $4

else
    echo "Kuch bhi"

fi
