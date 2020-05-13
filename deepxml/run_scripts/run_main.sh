#!/bin/bash
# $1 GPU DEIVCE ID
# $2 ABLATION TYPE
# $3 DATASET
# $4 VERSION
# eg. ./run_main.sh 0 DeepXML EURLex-4K 0 22
# eg. ./run_main.sh 0 DeepXML-fr EURLex-4K 0 22

export CUDA_VISIBLE_DEVICES=$1
model_type=$2
dataset=$3

source "../configs/${model_type}/${dataset}.sh"
version=$4
seed=$5
save_predictions=0

gen_aux_mapping () {
    # $1: train_feat_fname
    # $2: train_label_fname
    # $3: method
    # $4: threshold
    # $5: temp model data directory
    python3 ../tools/run_aux_mapping.py $1 $2 $3 $4 $5
}

merge_split_predictions () {
    # $1: fnames_predictions 
    # $2: fnames_mapping
    # $3: num_labels
    # $4: out_fname 
    echo "Merging predictions.."
    python3 ../tools/merge_split_predictions.py $1 $2 $3 $4 $5
}

clean_up(){
    echo "clean test train data copy"
    rm -rf ${trn_ft_file} ${trn_lbl_file} ${tst_ft_file} ${tst_lbl_file}
}

run_beta(){
    ./run_base.sh "evaluate" $1 $2 $3 $model_type $4 $5 $6 "${7} ${8} ${9}"
}


work_dir="${HOME}/scratch/Workspace"
data_dir="${work_dir}/data/${dataset}"
temp_model_data="${data_dir}/deepxml.aux/${aux_threshold}"

train_file="${data_dir}/train.txt"
test_file="${data_dir}/test.txt"
trn_ft_file="${data_dir}/trn_X_Xf.txt"
trn_lbl_file="${data_dir}/trn_X_Y.txt"
tst_ft_file="${data_dir}/tst_X_Xf.txt"
tst_lbl_file="${data_dir}/tst_X_Y.txt"
mkdir -p "${temp_model_data}"

convert() {
    perl ../tools/convert_format.pl $1 $2 $3
    perl ../tools/convert_format.pl $4 $5 $6
}

if [ ! -e "${trn_ft_file}" ]; then
    convert ${train_file} ${trn_ft_file} ${trn_lbl_file} ${test_file} ${tst_ft_file} ${tst_lbl_file}
fi


if [ ! -e "${temp_model_data}/aux_stats.json" ]
then
    gen_aux_mapping "${trn_ft_file}" "${trn_lbl_file}" $aux_method $aux_threshold $temp_model_data
else
    echo "Using old" "${temp_model_data}/aux_stats.json"
fi

run(){
    type=$1
    version=$2
    quantile=$3
    extra_params="${4}"
    learning_rate="lr_${quantile}"
    num_epochs="num_epochs_${quantile}"
    dlr_step="dlr_step_${quantile}"
    num_epochs="num_epochs_${quantile}"
    batch_size="batch_size_${quantile}"
    num_centriods="num_centroids_${quantile}"
    echo "Training ${quantile} .. with lr:" ${!learning_rate} "epochs:" ${!num_epochs}  
    ./run_"${type}".sh $dataset $version $quantile $use_post ${!learning_rate} $embedding_dims \
           ${!num_epochs} $dlr_factor ${!dlr_step} ${!batch_size} ${work_dir} \
           $model_type ${temp_model_data} ${topk} ${!num_centriods} \
           ${use_reranker} ${ns_method} ${seed} "${extra_params}"
}

results_dir="${work_dir}/results/$model_type/${dataset}/v_${version}"
models_dir="${work_dir}/models/$model_type/${dataset}/v_${version}"

if [ $num_splits -eq 0 ]; then
    echo "Not using any split to train."
    #run "full" $version "org" ""
    run "shortlist" $version "org" ""

    cp -r ${results_dir}/org/* ${results_dir}
    if [ $use_post -eq 1 ] || [] ; then
        run_beta $dataset $work_dir $version "test_predictions" $A $B $evaluation_type ${save_predictions} "0.1 0.2 0.5 0.6 0.75"
    else
        run_beta $dataset $work_dir $version "test_predictions" $A $B $evaluation_type ${save_predictions} -1
    fi

else
    run "full" $version "aux" "--aux_mapping ${temp_model_data}/aux_mapping.txt"
    run "shortlist" $version "org" ""
    cp "$results_dir/org/test_predictions_clf.npz" "$results_dir/test_predictions_clf.npz"
    cp "$results_dir/org/test_predictions_knn.npz" "$results_dir/test_predictions_knn.npz"
    echo "Evaluating with A/B: ${A}/${B}" $evaluation_type
    run_beta $dataset $work_dir $version "test_predictions" $A $B $evaluation_type ${save_predictions} "0.1 0.2 0.5 0.6 0.75"
    if [ $use_reranker -eq 1 ]
    then            
        mkdir -p "$models_dir/rnk"
        cp "$results_dir/org/test_predictions_reranker_clf.npz" "$models_dir/rnk/test_shortlist.npz"
        cp "$results_dir/org/train_predictions_clf.npz" "$models_dir/rnk/train_shortlist.npz"
        run "reranker" $version "rnk" ""
        ln -s "$results_dir/test_predictions_knn.npz" "$results_dir/test_predictions_reranker_knn.npz"
        cp "$results_dir/rnk/test_predictions_reranker_combined.npz" "$results_dir/test_predictions_reranker_clf.npz"
        run_beta $dataset $work_dir $version "test_predictions_reranker" $A $B $evaluation_type ${save_predictions} "0.1 0.2 0.5 0.6 0.75"
    fi
fi

#clean_up
