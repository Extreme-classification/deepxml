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
use_head_embeddings=0

if [ $use_head_embeddings -eq 1 ]
then
    echo "Using Head Embeddings"
    embedding_file="head_embeddings_${embedding_dims}d.npy"
else
    embedding_file="fasttextB_embeddings_${embedding_dims}d.npy"
fi

data_dir="${work_dir}/data"
stats=`python3 -c "import sys, json; print(json.load(open('${data_dir}/${dataset}/split_stats.json'))['${quantile}'])"` 
stats=($(echo $stats | tr ',' "\n"))
vocabulary_dims=${stats[0]}
num_labels=${stats[2]}
MODEL_NAME="deepxml_model"
extra_params=" --feature_indices ${data_dir}/${dataset}/features_split_${quantile}.txt --label_indices ${data_dir}/${dataset}/labels_split_${quantile}.txt"
if [ $quantile -eq -1 ]
then
    extra_params=""    
fi

TRAIN_PARAMS="--dataset ${dataset} \
                --data_dir=${work_dir}/data \
                --num_labels ${num_labels} \
                --vocabulary_dims ${vocabulary_dims} \
                --trans_method non_linear \
                --dropout 0.5 --optim Adam \
                --low_rank -1 \
                --efC 300 \
                --efS 300 \
                --num_clf_partitions 1\
                --lr $learning_rate \
                --use_residual \
                --embeddings $embedding_file \
                --embedding_dims ${embedding_dims} \
                --num_epochs $num_epochs \
                --dlr_factor $dlr_factor \
                --dlr_step $dlr_step \
                --batch_size $batch_size \
                --num_nbrs 300 \
                --M 100 \
                --normalize \
                --validate \
		        --val_feat_fname tst_X_Xf.txt \
                --val_label_fname tst_X_Y.txt \
                --ann_threads 12\
                --beta 0.5\
                --model_fname ${MODEL_NAME}${extra_params}"

if [ $use_post -eq 1 ]
then

    TRAIN_PARAMS_post="--dataset ${dataset} \
                --data_dir=${work_dir}/data \
                --num_labels ${num_labels} \
                --vocabulary_dims ${vocabulary_dims} \
                --trans_method non_linear \
                --dropout 0.5 --optim Adam \
                --low_rank -1 \
                --freeze_embeddings \
                --efC 300 \
                --efS 300 \
                --num_clf_partitions 2\
                --num_centroids 1 \
		        --use_residual \
                --lr $learning_rate \
                --embeddings $embedding_file \
                --embedding_dims ${embedding_dims} \
                --num_epochs $num_epochs \
                --dlr_factor $dlr_factor \
                --dlr_step $dlr_step \
                --batch_size $batch_size \
                --num_nbrs 300 \
                --M 100 \
                --use_shortlist \
                --normalize \
                --validate \
                --val_feat_fname tst_X_Xf.txt \
                --val_label_fname tst_X_Y.txt \
                --ann_threads 12\
                --beta 0.5\
                --model_fname ${MODEL_NAME}${extra_params}"

    PREDICT_PARAMS="--dataset ${dataset} \
                    --data_dir=${work_dir}/data \
                    --ts_feat_fname tst_X_Xf.txt \
                    --efS 300 \
                    --num_centroids 1 \
                    --num_nbrs 300 \
                    --ann_threads 12 \
                    --normalize \
                    --use_shortlist \
                    --model_fname ${MODEL_NAME} \
                    --batch_size 256 \
                    --out_fname predictions.txt \
                    --update_shortlist${extra_params}"

    EXTRACT_PARAMS="--dataset ${dataset} \
                    --data_dir=${work_dir}/data \
                    --normalize \
                    --use_shortlist \
                    --model_fname ${MODEL_NAME}\
                    --batch_size 512${extra_params}"

else
    PREDICT_PARAMS="--dataset ${dataset} \
                    --data_dir=${work_dir}/data \
                    --ts_feat_fname tst_X_Xf.txt \
                    --ts_label_fname tst_X_Y.txt \
                    --efS 300 \
                    --normalize \
                    --num_nbrs 300 \
                    --ann_threads 12\
                    --model_fname ${MODEL_NAME}\
                    --out_fname predictions.txt \
                    --batch_size 256${extra_params}"


    EXTRACT_PARAMS="--dataset ${dataset} \
                    --data_dir=${work_dir}/data \
                    --normalize \
                    --model_fname ${MODEL_NAME}\
                    --batch_size 512${extra_params}"

fi
docs=("trn" "tst")
cwd=$(pwd)

./run_base.sh "train" $dataset $work_dir $dir_version/$quantile "${TRAIN_PARAMS}"
./run_base.sh "predict" $dataset $work_dir $dir_version/$quantile "${PREDICT_PARAMS}"

if [ $use_post -eq 1 ]
then
    echo "Retraining with shortlist.."
   ./run_base.sh "retrain_w_shortlist" $dataset $work_dir $dir_version/$quantile "${TRAIN_PARAMS_post}"
fi

./run_base.sh "predict" $dataset $work_dir $dir_version/$quantile "${PREDICT_PARAMS}"
./run_base.sh "extract" $dataset $work_dir $dir_version/$quantile "${EXTRACT_PARAMS} --ts_fname 0 --out_fname export/wrd_emb"
if [ $quantile -gt -1 ]
then
    echo "Generating Head Embeddings"
    ./run_base.sh "gen_tail_emb" $dataset $work_dir $dir_version/$quantile "export/wrd_emb.npy" $quantile $embedding_dims
fi

for doc in ${docs[*]} 
do 
    ./run_base.sh "extract" $dataset $work_dir $dir_version/$quantile "${EXTRACT_PARAMS} --ts_feat_fname ${doc}_X_Xf.txt --ts_label_fname ${doc}_X_Y.txt --out_fname export/${doc}_emb"
    # ./run_base.sh "postprocess" $dataset $work_dir $dir_version/$quantile "export/${doc}_emb.npy" "${doc}"
done
