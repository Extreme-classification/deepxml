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
use_ensemble=${17}
use_head_embeddings=0
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
                --top_k ${topk} \
                --model_fname ${MODEL_NAME} ${extra_params} \
                --get_only knn clf"

TRAIN_PARAMS="--dlr_factor $dlr_factor \
            --dlr_step $dlr_step \
            --batch_size $batch_size \
            --num_clf_partitions 1\
            --trans_method ${current_working_dir}/shortlist.json \
            --dropout 0.5 
            --optim Adam \
            --ann_method ns \
            --model_method ns \
            --shortlist_method dynamic \
            --lr ${learning_rate} \
            --efS 500 \
            --normalize \
            --num_nbrs 500 \
            --use_shortlist \
            ${DEFAULT_PARAMS}"


if [ $use_post -eq 1 ]
then
    echo "Not tested yet!, Exiting..."
    exit
    # TRAIN_PARAMS_post="--trans_method ${current_working_dir}/shortlist.json \
    #             --dropout 0.5 --optim Adam \
    #             --efC 300 \
    #             --model_method shortlist \
    #             --efS 300 \
    #             --ann_method hnsw \
    #             --lr $learning_rate \
    #             --dlr_factor $dlr_factor \
    #             --dlr_step $dlr_step \
    #             --batch_size $batch_size \
    #             --num_nbrs 300 \
    #             --M 100 \
    #             --use_shortlist \
    #             --normalize \
    #             --ann_threads 12\
    #             --model_fname ${MODEL_NAME} ${DEFAULT_PARAMS}"

    # PREDICT_PARAMS="--efS 300 \
    #                 --num_centroids 1 \
    #                 --num_nbrs 300 \
    #                 --model_method shortlist \
    #                 --ann_threads 12 \
    #                 --normalize \
    #                 --use_shortlist \
    #                 --model_fname ${MODEL_NAME} \
    #                 --batch_size 256 \
    #                 --out_fname predictions.txt \
    #                 --update_shortlist ${extra_params}"

    # EXTRACT_PARAMS="--normalize \
    #                 --model_method shortlist \
    #                 --use_shortlist \
    #                 --model_fname ${MODEL_NAME}\
    #                 --batch_size 512 ${extra_params}"

else
    PREDICT_PARAMS="--model_method full \
                    --normalize \
                    --model_fname ${MODEL_NAME} \
                    --batch_size 256 ${DEFAULT_PARAMS} \
                    --pred_fname test_predictions"

    EXTRACT_PARAMS="--normalize \
                    --model_method full \
                    --model_fname ${MODEL_NAME}\
                    --batch_size 512 ${DEFAULT_PARAMS}"
fi

docs=("trn" "tst")
cwd=$(pwd)

./run_base.sh "train" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${TRAIN_PARAMS}"

if [ $use_post -eq 1 ]
then
    echo "Not tested yet!, Exiting..."
    exit
#     echo "Retraining with shortlist.."
#    ./run_base.sh "retrain_w_shortlist" $dataset $work_dir $dir_version/$quantile "${TRAIN_PARAMS_post}"
fi

./run_base.sh "predict" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${PREDICT_PARAMS}"

for doc in ${docs[*]}
do 
    ./run_base.sh "extract" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${EXTRACT_PARAMS}  --ts_feat_fname ${doc}_X_Xf.txt --ts_label_fname ${doc}_X_Y.txt --out_fname export/${doc}_emb"
    # ./run_base.sh "postprocess" $dataset $work_dir $dir_version/$version "export/${doc}_emb.npy" "${doc}"
done
