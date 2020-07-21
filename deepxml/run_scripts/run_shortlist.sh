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
use_aux_rep=1
current_working_dir=$(pwd)
data_dir="${work_dir}/data"
docs=("trn" "tst")


trn_ft_file="trn_X_Xf.txt"
trn_lbl_file="trn_X_Y.txt"
tst_ft_file="tst_X_Xf.txt"
tst_lbl_file="tst_X_Y.txt"

extra_params="${extra_params} --feature_type sparse --normalize"


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
                --tr_feat_fname ${trn_ft_file} \
                --tr_label_fname ${trn_lbl_file} \
		        --val_feat_fname ${tst_ft_file} \
                --val_label_fname ${tst_lbl_file} \
                --ts_feat_fname ${tst_ft_file} \
                --ts_label_fname ${tst_lbl_file} \
                --freeze_embeddings \
                --top_k ${topk} \
                --seed ${seed} \
                --num_centroids ${num_centroids} \
                --model_fname ${MODEL_NAME} ${extra_params} \
                --get_only knn clf"

TRAIN_PARAMS="--dlr_factor $dlr_factor \
            --dlr_step $dlr_step \
            --batch_size $batch_size \
            --num_clf_partitions 1\
            --trans_method ${current_working_dir}/shortlist.json \
            --dropout 0.5 
            --optim Adam \
            --model_method shortlist \
            --shortlist_method hybrid \
            --lr ${learning_rate} \
            --efS 500 \
            --num_nbrs 500 \
            --efC 300 \
            --M 100 \
            --use_shortlist \
            --validate \
		    --ann_threads 12 \
            --beta 0.5 \
            --update_shortlist \
            --use_coarse_for_shorty \
            --retrain_hnsw_after $(($num_epochs+3)) \
            ${DEFAULT_PARAMS}"

PREDICT_PARAMS="--efS 500 \
                --num_nbrs 500 \
                --model_method shortlist \
                --ann_threads 12\
                --use_shortlist \
                --batch_size 256 \
                --use_coarse_for_shorty \
                --beta 0.5 ${extra_params} \
                --out_fname predictions.txt \
                --pred_fname test_predictions \
                --update_shortlist \
                ${DEFAULT_PARAMS}"


PREDICT_PARAMS_train="--efS 500 \
                    --num_nbrs 500 \
                    --model_method shortlist \
                    --ann_threads 12\
                    --use_shortlist \
                    --batch_size 256 \
                    --use_coarse_for_shorty \
                    --beta 0.5 ${extra_params}\
                    --out_fname predictions.txt \
                    --pred_fname train_predictions \
                    --update_shortlist \
                    ${DEFAULT_PARAMS} \
                    --ts_feat_fname ${trn_ft_file} \
                    --ts_label_fname ${trn_lbl_file}"

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

for doc in ${docs[*]}
do 
    ./run_base.sh "extract" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${EXTRACT_PARAMS} --ts_feat_fname ${doc}_X_Xf.txt --out_fname export/${doc}_emb"
done

