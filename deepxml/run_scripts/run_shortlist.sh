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
use_aux_rep=1
extract_embeddings=1
current_working_dir=$(pwd)
data_dir="${work_dir}/data"
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

stats=`python3 -c "import sys, json; print(json.load(open('${temp_model_data}/aux_stats.json'))['${quantile}'])"` 
stats=($(echo $stats | tr ',' "\n"))
vocabulary_dims=${stats[0]}
num_labels=${stats[2]}

if [ $use_aux_rep -eq 1 ]
then
    echo -e "\nUsing parameters from auxilliary task."
    extra_params="${extra_params} --load_intermediate"
else
    echo -e "\nUsing pre-trained embeddings."
    embedding_file="fasttextB_embeddings_${embedding_dims}d.npy"
    extra_params="${extra_params} --embeddings ${embedding_file}"
fi

DEFAULT_PARAMS="--dataset ${dataset} \
                --data_dir=${work_dir}/data \
                --num_labels ${num_labels} \
                --vocabulary_dims ${vocabulary_dims} \
                --ns_method ${ns_method} \
                --embedding_dims ${embedding_dims} \
                --num_epochs $num_epochs \
                --trn_feat_fname ${trn_feat_fname} \
                --trn_label_fname ${trn_label_fname} \
		        --val_feat_fname ${val_feat_fname} \
                --val_label_fname ${val_label_fname} \
                --freeze_intermediate \
                --top_k ${topk} \
                --seed ${seed} \
                --num_centroids ${num_centroids} \
                --model_fname ${MODEL_NAME} ${extra_params} \
                --get_only knn clf"

TRAIN_PARAMS="--dlr_factor $dlr_factor \
            --dlr_step $dlr_step \
            --batch_size $batch_size \
            --num_clf_partitions 1\
            --trans_method ${farch} \
            --dropout 0.5 
            --optim Adam \
            --model_method shortlist \
            --shortlist_method hybrid \
            --lr ${learning_rate} \
            --efS 400 \
            --num_nbrs 500 \
            --efC 300 \
            --M 100 \
            --use_shortlist \
            --validate \
		    --ann_threads 12 \
            --beta 0.5 \
            --update_shortlist \
            --use_intermediate_for_shorty \
            --retrain_hnsw_after $(($num_epochs+3)) \
            ${DEFAULT_PARAMS}"

PREDICT_PARAMS="--efS 400 \
                --num_nbrs 500 \
                --tst_feat_fname ${tst_feat_fname} \
                --tst_label_fname ${tst_label_fname} \
                --model_method shortlist \
                --ann_threads 12\
                --use_shortlist \
                --batch_size 256 \
                --tst_feat_fname ${tst_feat_fname} \
                --tst_label_fname ${tst_label_fname} \
                --use_intermediate_for_shorty \
                --beta 0.5 ${extra_params} \
                --out_fname predictions.txt \
                --pred_fname test_predictions \
                --update_shortlist \
                ${DEFAULT_PARAMS}"


PREDICT_PARAMS_train="--efS 400 \
                    --num_nbrs 500 \
                    --model_method shortlist \
                    --ann_threads 12\
                    --use_shortlist \
                    --batch_size 256 \
                    --use_intermediate_for_shorty \
                    --beta 0.5 ${extra_params}\
                    --out_fname predictions.txt \
                    --pred_fname train_predictions \
                    --update_shortlist \
                    ${DEFAULT_PARAMS} \
                    --tst_feat_fname ${trn_feat_fname} \
                    --tst_label_fname ${trn_label_fname}"

EXTRACT_PARAMS="--dataset ${dataset} \
                --data_dir=${work_dir}/data \
                --use_shortlist \
                --model_method shortlist \
                --ns_method ${ns_method} \
                --model_fname ${MODEL_NAME} \
                --batch_size 512 ${extra_params}"

./run_base.sh "train" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${TRAIN_PARAMS}"
./run_base.sh "predict" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${PREDICT_PARAMS}"

if [ $use_reranker -eq 1 ]
then
    echo -e "\nFetching data for reranker"
    ./run_base.sh "predict" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${PREDICT_PARAMS_train} --get_only clf"
fi

if [ $extract_embeddings -eq 1 ]
then
    ./run_base.sh "extract" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${EXTRACT_PARAMS} --tst_feat_fname "${trn_feat_fname}" --out_fname export/trn_embedding"
    ./run_base.sh "extract" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${EXTRACT_PARAMS} --tst_feat_fname "${tst_feat_fname}" --out_fname export/tst_embedding"
fi