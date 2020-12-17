#!/bin/bash
# $1 GPU Device ID
# $2 Model Type (DeepXML/DeepXML-fr etc.)
# $3 Architecture type (Astec etc.)
# $4 Dataset
# $5 version
# $6 seed
# eg. ./run_main.sh 0 DeepXML Astec EURLex-4K 0 22
# eg. ./run_main.sh 0 DeepXML-fr Astec EURLex-4K 0 22

export CUDA_VISIBLE_DEVICES=$1
model_type=$2
arch_type=$3
dataset=$4
feature_type='sparse' # default
source "../configs/${model_type}/${dataset}.sh"
version=$5
seed=$6
save_predictions=1
betas="0.1 0.2 0.5 0.6 0.75 0.8 0.9"

gen_aux_mapping () {
    python3 ../tools/run_aux_mapping.py ${trn_ft_file} ${trn_lbl_file} ${aux_method} ${aux_threshold} ${seed} ${temp_model_data}
}

clean_up(){
    echo "clean test train data copy"
    rm -rf ${trn_ft_file} ${trn_lbl_file} ${tst_ft_file} ${tst_lbl_file}
}

evaluate() {
    # $1 fname
    # $2 beta
    ./run_base.sh "evaluate" ${dataset} "${work_dir}" ${version} ${model_type} "${1}" $A $B "${save_predictions} ${2}"
}

work_dir=$(cd ../../../../ && pwd)
data_dir="${work_dir}/data/${dataset}"
temp_model_data="${data_dir}/deepxml.aux/${aux_threshold}.${seed}"

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
    gen_aux_mapping
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
    echo -e "\nTraining (${arch_type}) ${quantile} .. with lr:" ${!learning_rate} " & epochs:" ${!num_epochs}  
    ./run_"${type}".sh ${arch_type} $dataset ${feature_type} $version $quantile \
        $use_post ${!learning_rate} $embedding_dims \
        ${!num_epochs} $dlr_factor ${!dlr_step} ${!batch_size} ${work_dir} \
        $model_type ${temp_model_data} ${topk} ${!num_centriods} \
        ${use_reranker} ${ns_method} ${seed} "${extra_params}"
}

results_dir="${work_dir}/results/$model_type/${dataset}/v_${version}"
models_dir="${work_dir}/models/$model_type/${dataset}/v_${version}"

if [ ${model_type} == "DeepXML-OVA" ]; then
    echo -e "\n---Running DeepXML-OVA, i.e., DeepXML with OVA classifier---\n"
    run "full" $version "org" ""
    cp -r ${results_dir}/org/* ${results_dir}
    echo -e "\nEvaluating with A/B: ${A}/${B}"
    if [ $use_post -eq 1 ]; then
        evaluate "test_predictions" "${betas}"
    else
        evaluate "test_predictions" "-1"
    fi
elif [ ${model_type} == "DeepXML-ANNS" ]; then
    echo -e "\n---Running DeepXML-ANNS, i.e., DeepXML with ANNS based classifier---\n"
    run "shortlist" $version "org" ""
    cp -r ${results_dir}/org/* ${results_dir}
    echo -e "\nEvaluating with A/B: ${A}/${B}"
    evaluate "test_predictions" "${betas}"
else
    echo -e "\n---Running DeepXML---\n"
    run "full" $version "aux" "--aux_mapping ${temp_model_data}/aux_mapping.txt"
    run "shortlist" $version "org" ""
    cp "$results_dir/org/test_predictions_clf.npz" "$results_dir/test_predictions_clf.npz"
    cp "$results_dir/org/test_predictions_knn.npz" "$results_dir/test_predictions_knn.npz"
    echo -e "\nEvaluating base classifier with A/B: ${A}/${B}"
    evaluate "test_predictions" "${betas}"
    if [ $use_reranker -eq 1 ]; then            
        mkdir -p "$models_dir/rnk"
        cp "$results_dir/org/test_predictions_clf.npz" "$models_dir/rnk/test_shortlist.npz"
        cp "$results_dir/org/train_predictions_clf.npz" "$models_dir/rnk/train_shortlist.npz"
        run "reranker" $version "rnk" ""
        ln -s "$results_dir/test_predictions_knn.npz" "$results_dir/test_predictions_reranker_knn.npz"
        cp "$results_dir/rnk/test_predictions_reranker_combined.npz" "$results_dir/test_predictions_reranker_clf.npz"
        echo -e "\nEvaluating re-ranker with with A/B: ${A}/${B}\n"
        evaluate "test_predictions_reranker" "${betas}"
    fi
fi

#clean_up
