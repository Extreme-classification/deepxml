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
seed=${20}
use_aux_rep=0
current_working_dir=$(pwd)
data_dir="${work_dir}/data"

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

# Do not remove invalid labels
stats=`python3 -c "import sys, json; print(json.load(open('${temp_model_data}/aux_stats.json'))['org'])"`
stats=($(echo $stats | tr ',' "\n"))
vocabulary_dims=${stats[0]}
num_labels=${stats[1]}

if [ $use_aux_rep -eq 1 ]
then
    echo -e "\nUsing parameters from auxilliary task."
    extra_params="${extra_params} --load_intermediate"
else
    echo -e "\nUsing pre-trained embeddings."
    embedding_file="fasttextB_embeddings_${embedding_dims}d.npy"
fi

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
                --seed ${seed} \
                --label_padding_index ${num_labels} \
                --top_k ${topk} \
                --model_fname ${MODEL_NAME} ${extra_params} \
                --get_only combined \
                --use_pretrained_shortlist"

TRAIN_PARAMS="--dlr_factor $dlr_factor \
            --dlr_step $dlr_step \
            --batch_size $batch_size \
            --trans_method ${farch} \
            --dropout 0.5 
            --optim Adam \
            --keep_invalid \
            --model_method reranker \
            --shortlist_method static \
            --lr $learning_rate \
            --num_nbrs $topk \
            --use_shortlist \
            --validate \
		    --ann_threads 12 \
            --beta 0.5 \
            --retrain_hnsw_after $(($num_epochs+3)) \
            ${DEFAULT_PARAMS}"

PREDICT_PARAMS="--tst_feat_fname ${tst_feat_fname} \
                --tst_label_fname ${tst_label_fname} \
                --model_method reranker \
                --shortlist_method reranker \
                --num_nbrs $topk \
                --ann_threads 12 \
                --keep_invalid \
                --use_shortlist \
                --batch_size 256 \
                --pred_fname test_predictions_reranker \
                --update_shortlist \
                ${DEFAULT_PARAMS}"


EXTRACT_PARAMS="--dataset ${dataset} \
                --data_dir=${work_dir}/data \
                --use_shortlist \
                --model_method shortlist \
                --model_fname ${MODEL_NAME}\
                --batch_size 512 ${extra_params}"

./run_base.sh "train" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${TRAIN_PARAMS}"
./run_base.sh "predict" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${PREDICT_PARAMS}"
