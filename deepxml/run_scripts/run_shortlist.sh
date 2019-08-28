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
use_head_embeddings=1
current_working_dir=$(pwd)
data_dir="${work_dir}/data"
docs=("trn" "tst")


stats=`python3 -c "import sys, json; print(json.load(open('${data_dir}/${dataset}/${temp_model_data}/${split_threhold}/split_stats.json'))['${quantile}'])"` 
stats=($(echo $stats | tr ',' "\n"))
vocabulary_dims=${stats[0]}
num_labels=${stats[2]}
echo $num_labels $(wc -l ${data_dir}/${dataset}/${temp_model_data}/${split_threhold}/labels_split_${quantile}.txt)
extra_params="--feature_indices ${data_dir}/${dataset}/${temp_model_data}/${split_threhold}/features_split_${quantile}.txt \
                --label_indices ${data_dir}/${dataset}/${temp_model_data}/${split_threhold}/labels_split_${quantile}.txt"

if [ $quantile -eq -1 ]
then
    extra_params=""    
fi

<<<<<<< HEAD
if [ $use_head_embeddings -eq 1 ]
then
    echo "Using Head Embeddings"
    embedding_file="head_embeddings_${embedding_dims}d.npy"
    extra_params="${extra_params} --use_head_embeddings"
else
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
                --model_fname ${MODEL_NAME} ${extra_params}"

TRAIN_PARAMS="--dlr_factor $dlr_factor \
            --dlr_step $dlr_step \
            --batch_size $batch_size \
            --num_clf_partitions 1\
=======
current_working_dir=$(pwd)

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
>>>>>>> c8451fcf6d4a49155b5e6fff1cab93fff8586e83
            --trans_method ${current_working_dir}/shortlist.json \
            --dropout 0.5 
            --optim Adam \
            --model_method shortlist \
            --efS 300 \
            --normalize \
            --num_nbrs 300 \
            --efC 300 \
            --M 100 \
            --use_shortlist \
            --validate \
		    --ann_threads 12 \
            --beta 0.5 \
            --update_shortlist \
            --retrain_hnsw_after $(($num_epochs+3)) \
            ${DEFAULT_PARAMS}"

PREDICT_PARAMS="--efS 300 \
                --num_nbrs 300 \
                --model_method shortlist \
                --ann_threads 12\
                --normalize \
                --use_shortlist \
                --batch_size 256 \
                --beta 0.5 ${extra_params}\
                --out_fname predictions.txt \
                --use_coarse_for_shorty \
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
./run_base.sh "predict" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${PREDICT_PARAMS}"
./run_base.sh "extract" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${EXTRACT_PARAMS} --ts_feat_fname 0 --out_fname export/wrd_emb"
for doc in ${docs[*]}
do 
    ./run_base.sh "extract" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${EXTRACT_PARAMS}  --ts_feat_fname ${doc}_X_Xf.txt --ts_label_fname ${doc}_X_Y.txt --out_fname export/${doc}_emb"
    # ./run_base.sh "postprocess" $dataset $work_dir $dir_version/$version "export/${doc}_emb.npy" "${doc}"
done

