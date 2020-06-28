#!/bin/bash
dataset=$1
dir_version=$2
quantile=$3
use_post=$4
learning_rate=${5}
embedding_dims=$6
num_epochs=$7
dlr_factor=$8
dlr_step=${9}
batch_size=${10}
work_dir=${11}
MODEL_NAME="${12}"
temp_model_data="${13}"
topk="${14}"
seed=${18}
use_aux_embeddings=0
current_working_dir=$(pwd)
data_dir="${work_dir}/data"
docs=("trn" "tst")


# Do not remove invalid labels
stats=`python3 -c "import sys, json; print(json.load(open('${temp_model_data}/aux_stats.json'))['org'])"`
stats=($(echo $stats | tr ',' "\n"))
vocabulary_dims=${stats[0]}
num_labels=${stats[1]}

if [ $use_aux_embeddings -eq 1 ]
then
    echo -e "\nUsing embeddings from auxilliary task."
    embedding_file="aux_embeddings_${embedding_dims}d.npy"
    extra_params="${extra_params} --use_aux_embeddings"
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
                --tr_feat_fname trn_X_Xf.txt \
                --tr_label_fname trn_X_Y.txt \
		        --val_feat_fname tst_X_Xf.txt \
                --val_label_fname tst_X_Y.txt \
                --ts_feat_fname tst_X_Xf.txt \
                --ts_label_fname tst_X_Y.txt \
                --seed ${seed} \
                --label_padding_index ${num_labels} \
                --top_k ${topk} \
                --model_fname ${MODEL_NAME} ${extra_params} \
                --get_only combined \
                --use_pretrained_shortlist"

TRAIN_PARAMS="--dlr_factor $dlr_factor \
            --dlr_step $dlr_step \
            --batch_size $batch_size \
            --num_clf_partitions 1 \
            --trans_method ${current_working_dir}/reranker.json \
            --dropout 0.5 
            --optim Adam \
            --keep_invalid \
            --model_method reranker \
            --shortlist_method static \
            --lr $learning_rate \
            --efS 300 \
            --normalize \
            --num_nbrs $topk \
            --efC 300 \
            --M 100 \
            --use_shortlist \
            --validate \
		    --ann_threads 12 \
            --beta 0.5 \
            --retrain_hnsw_after $(($num_epochs+3)) \
            ${DEFAULT_PARAMS}"

PREDICT_PARAMS_TEST="--efS 300 \
                    --model_method reranker \
                    --shortlist_method reranker \
                    --num_centroids 1 \
                    --num_nbrs $topk \
                    --ann_threads 12 \
                    --normalize \
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
                --normalize \
                --model_fname ${MODEL_NAME}\
                --batch_size 512 ${extra_params}"

./run_base.sh "train" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${TRAIN_PARAMS}"
./run_base.sh "predict" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${PREDICT_PARAMS_TEST}"
