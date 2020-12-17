#!/bin/bash
arch=$1
dataset=$2
feature_type=$3
dir_version=$4
quantile=$5
use_post=$6
learning_rate=${7}
embedding_dims=$8
num_epochs=$9
dlr_factor=${10}
dlr_step=${11}
batch_size=${12}
work_dir=${13}
MODEL_NAME="${14}"
temp_model_data="${15}"
topk=${16}
num_centroids=${17}
use_reranker=${18}
ns_method=${19}
seed=${20}
extra_params="${21}"
extract_embeddings=1
data_dir="${work_dir}/data"
current_working_dir=$(pwd)
docs=("trn" "tst")

extra_params="${extra_params} --feature_type ${feature_type}"

trn_label_fname="trn_X_Y.txt"
val_label_fname="tst_X_Y.txt"
tst_label_fname="tst_X_Y.txt"
if [ $feature_type == "sparse" ]
then
    trn_feat_fname="trn_X_Xf.txt"
    val_feat_fname="tst_X_Xf.txt"
    tst_feat_fname="tst_X_Xf.txt"
    extra_params="${extra_params} --normalize"
elif [ $feature_type == "sequential" ]
then
    trn_feat_fname="trn_X_Xf.pkl"
    val_feat_fname="tst_X_Xf.pkl"
    tst_feat_fname="tst_X_Xf.pkl"
    extra_params="${extra_params} --normalize"
else
    echo -e "Not yet implemented. Exiting.."
    exit
fi

farch="${current_working_dir}/${arch,,}.json"

echo -e "\nUsing pre-trained embeddings."
embedding_file="fasttextB_embeddings_${embedding_dims}d.npy"

stats=`python3 -c "import sys, json; print(json.load(open('${temp_model_data}/aux_stats.json'))['${quantile}'])"` 
stats=($(echo $stats | tr ',' "\n"))
vocabulary_dims=${stats[0]}
num_labels=${stats[2]}

DEFAULT_PARAMS="--dataset ${dataset} \
                --data_dir=${work_dir}/data \
                --num_labels ${num_labels} \
                --vocabulary_dims ${vocabulary_dims} \
                --embeddings $embedding_file \
                --embedding_dims ${embedding_dims} \
                --num_epochs $num_epochs \
                --trn_feat_fname ${trn_feat_fname} \
                --trn_label_fname ${trn_label_fname} \
		        --val_feat_fname ${val_feat_fname} \
                --val_label_fname ${val_label_fname} \
                --top_k $topk \
                --seed ${seed} \
                --model_fname ${MODEL_NAME} ${extra_params} \
                --get_only knn clf"

TRAIN_PARAMS="  --trans_method ${farch} \
                --dropout 0.5 --optim Adam \
                --num_clf_partitions 1\
                --lr $learning_rate \
                --model_method full \
                --dlr_factor $dlr_factor \
                --dlr_step $dlr_step \
                --batch_size $batch_size \
                --validate \
                --save_intermediate \
                ${DEFAULT_PARAMS}"
PREDICT_PARAMS="--model_method full \
                --model_fname ${MODEL_NAME}\
                --tst_feat_fname ${tst_feat_fname} \
                --tst_label_fname ${tst_label_fname} \
                --pred_fname test_predictions \
                --out_fname predictions.txt \
                --batch_size 256 \
                ${DEFAULT_PARAMS}"

EXTRACT_PARAMS="--dataset ${dataset} \
                --data_dir=${work_dir}/data \
                --model_method full \
                --model_fname ${MODEL_NAME}\
                --batch_size 512 ${extra_params}"


./run_base.sh "train" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${TRAIN_PARAMS}"
./run_base.sh "predict" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${PREDICT_PARAMS}"

if [ $extract_embeddings -eq 1 ]
then
    ./run_base.sh "extract" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${EXTRACT_PARAMS} --tst_feat_fname "${trn_feat_fname}" --out_fname export/trn_embedding"
    ./run_base.sh "extract" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${EXTRACT_PARAMS} --tst_feat_fname "${tst_feat_fname}" --out_fname export/tst_embedding"
fi