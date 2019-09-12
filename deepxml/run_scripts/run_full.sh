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
split_threhold="${14}"
topk=${15}
num_centroids=${16}
echo $num_centroids
use_head_embeddings=0
data_dir="${work_dir}/data"
current_working_dir=$(pwd)
docs=("trn" "tst")

if [ $use_head_embeddings -eq 1 ]
then
    echo "Using Head Embeddings"
    embedding_file="head_embeddings_${embedding_dims}d.npy"
else
    embedding_file="fasttextB_embeddings_${embedding_dims}d.npy"
fi

stats=`python3 -c "import sys, json; print(json.load(open('${data_dir}/${dataset}/${temp_model_data}/${split_threhold}/split_stats.json'))['${quantile}'])"` 
stats=($(echo $stats | tr ',' "\n"))
vocabulary_dims=${stats[0]}
num_labels=${stats[2]}

extra_params="--feature_indices ${data_dir}/${dataset}/${temp_model_data}/${split_threhold}/features_split_${quantile}.txt \
                --label_indices ${data_dir}/${dataset}/${temp_model_data}/${split_threhold}/labels_split_${quantile}.txt"

if [ $quantile -eq -1 ]
then
    extra_params=""    
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
                --top_k $topk \
                --model_fname ${MODEL_NAME} ${extra_params}"

TRAIN_PARAMS="  --trans_method ${current_working_dir}/full.json \
                --dropout 0.5 --optim Adam \
                --num_clf_partitions 1\
                --lr $learning_rate \
                --model_method full \
                --dlr_factor $dlr_factor \
                --dlr_step $dlr_step \
                --batch_size $batch_size \
                --normalize \
                --validate \
                ${DEFAULT_PARAMS}"

if [ $use_post -eq 1 ]
then

    TRAIN_PARAMS_post="--trans_method  ${current_working_dir}/full.json \
                --dropout 0.5 --optim Adam \
                --low_rank -1 \
                --freeze_embeddings \
                --efC 300 \
                --efS 300 \
                --num_clf_partitions 1\
                --model_method full \
                --num_centroids ${num_centroids} \
		        --lr $learning_rate \
                --dlr_factor $dlr_factor \
                --dlr_step $dlr_step \
                --batch_size $batch_size \
                --num_nbrs 300 \
                --M 100 \
                --use_shortlist \
                --normalize \
                --validate \
                --ann_threads 12\
                --beta 0.5\
                ${DEFAULT_PARAMS}"

    PREDICT_PARAMS="--efS 300 \
                    --model_method shortlist \
                    --num_centroids ${num_centroids} \
                    --num_nbrs 300 \
                    --ann_threads 12 \
                    --normalize \
                    --use_shortlist \
                    --batch_size 256 \
                    --pred_fname test_predictions \
                    --out_fname test_predictions.txt \
                    --update_shortlist \
                    ${DEFAULT_PARAMS}"
    
    PREDICT_PARAMS_train="--efS 300 \
                        --model_method shortlist \
                        --num_centroids ${num_centroids} \
                        --num_nbrs 300 \
                        --ann_threads 12 \
                        --normalize \
                        --use_shortlist \
                        --batch_size 256 \
                        --pred_fname train_predictions \
                        --out_fname train_predictions.txt \
                        --update_shortlist \
                        ${DEFAULT_PARAMS} \
                        --ts_feat_fname trn_X_Xf.txt \
                        --ts_label_fname trn_X_Y.txt"

    EXTRACT_PARAMS="--dataset ${dataset} \
                    --data_dir=${work_dir}/data \
                    --normalize \
                    --model_method shortlist \
                    --use_shortlist \
                    --model_fname ${MODEL_NAME}\
                    --batch_size 512 ${extra_params}"

else
    PREDICT_PARAMS="--model_method full \
                    --normalize \
                    --model_fname ${MODEL_NAME}\
                    --pred_fname test_predictions \
                    --out_fname predictions.txt \
                    --batch_size 256 \
                    ${DEFAULT_PARAMS}"

    EXTRACT_PARAMS="--dataset ${dataset} \
                    --data_dir=${work_dir}/data \
                    --normalize \
                    --model_method full \
                    --model_fname ${MODEL_NAME}\
                    --batch_size 512 ${extra_params}"

fi

./run_base.sh "train" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${TRAIN_PARAMS}"

if [ $use_post -eq 1 ]
then
    echo "Retraining with shortlist.."
   ./run_base.sh "retrain_w_shortlist" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${TRAIN_PARAMS_post}"
fi
./run_base.sh "predict" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${PREDICT_PARAMS}"
./run_base.sh "predict" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${PREDICT_PARAMS_train}"

./run_base.sh "extract" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${EXTRACT_PARAMS} --ts_feat_fname 0 --out_fname export/wrd_emb"

if [ $quantile -gt -1 ]
then
    echo "Generating Head Embeddings"
    ./run_base.sh "gen_tail_emb" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "export/wrd_emb.npy" $quantile $embedding_dims "${temp_model_data}/${split_threhold}"
fi

for doc in ${docs[*]} 
do 
    ./run_base.sh "extract" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${EXTRACT_PARAMS} --ts_feat_fname ${doc}_X_Xf.txt --ts_label_fname ${doc}_X_Y.txt --out_fname export/${doc}_emb"
    # ./run_base.sh "postprocess" $dataset $work_dir $dir_version/$quantile "export/${doc}_emb.npy" "${doc}"
done
