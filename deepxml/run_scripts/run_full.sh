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
topk=${14}
num_centroids=${15}
use_reranker=${16}
ns_method=${17}
seed=${18}
extra_params="${19}"
data_dir="${work_dir}/data"
current_working_dir=$(pwd)
docs=("trn" "tst")

trn_ft_file="trn_X_Xf.txt"
trn_lbl_file="trn_X_Y.txt"
tst_ft_file="tst_X_Xf.txt"
tst_lbl_file="tst_X_Y.txt"

extra_params="${extra_params} --normalize --feature_type sparse"

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
                --tr_feat_fname ${trn_ft_file} \
                --tr_label_fname ${trn_lbl_file} \
		        --val_feat_fname ${tst_ft_file} \
                --val_label_fname ${tst_lbl_file} \
                --ts_feat_fname ${tst_ft_file} \
                --ts_label_fname ${tst_lbl_file} \
                --top_k $topk \
                --seed ${seed} \
                --model_fname ${MODEL_NAME} ${extra_params} \
                --get_only knn clf"

TRAIN_PARAMS="  --trans_method ${current_working_dir}/full.json \
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


if [ $use_post -eq 1 ]
then
    TRAIN_PARAMS_post="--trans_method  ${current_working_dir}/full.json \
                --dropout 0.5 --optim Adam \
                --freeze_intermediate \
                --efC 300 \
                --efS 300 \
                --ns_method ${ns_method} \
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
                --validate \
                --ann_threads 24\
                --beta 0.5\
                ${DEFAULT_PARAMS}"

    PREDICT_PARAMS="--efS 300 \
                    --model_method shortlist \
                    --num_centroids ${num_centroids} \
                    --num_nbrs 300 \
                    --ns_method ${ns_method} \
                    --ann_threads 24 \
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
                        --ann_threads 24 \
                        --ns_method ${ns_method} \
                        --use_shortlist \
                        --batch_size 256 \
                        --pred_fname train_predictions \
                        --out_fname train_predictions.txt \
                        --update_shortlist \
                        ${DEFAULT_PARAMS} \
                        --ts_feat_fname ${trn_ft_file} \
                        --ts_label_fname ${trn_lbl_file} \
                        --get_only clf"

    EXTRACT_PARAMS="--dataset ${dataset} \
                    --data_dir=${work_dir}/data \
                    --ns_method ${ns_method} \
                    --model_method shortlist \
                    --use_shortlist \
                    --model_fname ${MODEL_NAME}\
                    --batch_size 512 ${extra_params}"

else
    PREDICT_PARAMS="--model_method full \
                    --model_fname ${MODEL_NAME}\
                    --pred_fname test_predictions \
                    --out_fname predictions.txt \
                    --batch_size 256 \
                    ${DEFAULT_PARAMS}"

    EXTRACT_PARAMS="--dataset ${dataset} \
                    --data_dir=${work_dir}/data \
                    --model_method full \
                    --model_fname ${MODEL_NAME}\
                    --batch_size 512 ${extra_params}"

fi

./run_base.sh "train" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${TRAIN_PARAMS}"

if [ $use_post -eq 1 ]
then
   echo -e "\nRetraining with shortlist.."
  ./run_base.sh "retrain_w_shortlist" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${TRAIN_PARAMS_post}"
fi

./run_base.sh "predict" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${PREDICT_PARAMS}"

for doc in ${docs[*]} 
do 
    ./run_base.sh "extract" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${EXTRACT_PARAMS} --ts_feat_fname ${doc}_X_Xf.txt --out_fname export/${doc}_emb"
done
