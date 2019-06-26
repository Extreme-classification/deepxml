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
use_head_embeddings=1

data_dir="${work_dir}/data"
extra_params="--feature_indices ${data_dir}/${dataset}/features_split_${quantile}.txt --label_indices ${data_dir}/${dataset}/labels_split_${quantile}.txt"
if [ $use_head_embeddings -eq 1 ]
then
    echo "Using Head Embeddings"
    embedding_file="head_embeddings_${embedding_dims}d.npy"
    extra_params="$extra_params --use_head_embeddings"
else
    embedding_file="fasttextB_embeddings_${embedding_dims}d.npy"
fi


stats=`python3 -c "import sys, json; print(json.load(open('${data_dir}/${dataset}/split_stats.json'))['${quantile}'])"` 
stats=($(echo $stats | tr ',' "\n"))
vocabulary_dims=${stats[0]}
num_labels=${stats[2]}
MODEL_NAME="deepxml_model"

if [ $quantile -eq -1 ]
then
    extra_params=""    
fi

echo $embedding_file
TRAIN_PARAMS="--lr $learning_rate \
            --embeddings $embedding_file \
            --embedding_dims $embedding_dims \
            --num_epochs $num_epochs \
            --dlr_factor $dlr_factor \
            --dlr_step $dlr_step \
            --batch_size $batch_size \
            --num_clf_partitions 1\
            --dataset ${dataset} \
            --data_dir=${work_dir}/data \
            --num_labels ${num_labels} \
            --vocabulary_dims ${vocabulary_dims} \
            --trans_method non_linear \
            --dropout 0.5 
            --optim Adam \
            --low_rank -1 \
            --efS 300 \
            --normalize \
            --num_nbrs 300 \
            --efC 300 \
            --M 100 \
            --use_shortlist \
            --validate \
		    --val_feat_fname tst_X_Xf.txt \
            --val_label_fname tst_X_Y.txt \
            --ann_threads 12 \
            --beta 0.5 \
            --update_shortlist \
            --model_fname ${MODEL_NAME} ${extra_params} \
            --retrain_hnsw_after $(($num_epochs+3))"
            # --retrain_hnsw_after $(($num_epochs+3))"
            # --retrain_hnsw_after 5"

PREDICT_PARAMS="--dataset ${dataset} \
                --data_dir=${work_dir}/data \
    		    --ts_feat_fname tst_X_Xf.txt \
                --ts_label_fname tst_X_Y.txt \
                --efS 300 \
                --num_nbrs 300 \
                --ann_threads 12\
                --normalize \
                --use_shortlist \
                --model_fname ${MODEL_NAME}\
                --batch_size 256 \
                --beta 0.5 ${extra_params}\
                --out_fname predictions.txt"
                #--update_shortlist"

EXTRACT_PARAMS="--dataset ${dataset} \
                --data_dir=${work_dir}/data \
                --use_shortlist \
                --normalize \
                --model_fname ${MODEL_NAME}\
                --batch_size 512 ${extra_params}"

docs=("trn" "tst")
cwd=$(pwd)
./run_base.sh "train" $dataset $work_dir $dir_version/$quantile "${TRAIN_PARAMS}"
./run_base.sh "predict" $dataset $work_dir $dir_version/$quantile "${PREDICT_PARAMS}"
# ./run_base.sh "extract" $dataset $work_dir $dir_version/$version "${EXTRACT_PARAMS} --ts_fname 0 --out_fname export/wrd_emb"
for doc in ${docs[*]}
do 
    ./run_base.sh "extract" $dataset $work_dir $dir_version/$quantile "${EXTRACT_PARAMS}  --ts_feat_fname ${doc}_X_Xf.txt --ts_label_fname ${doc}_X_Y.txt --out_fname export/${doc}_emb"
    # ./run_base.sh "postprocess" $dataset $work_dir $dir_version/$version "export/${doc}_emb.npy" "${doc}"
done

# source deactivate
