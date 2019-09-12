#!/bin/bash
# $1 GPU DEIVCE ID
# $2 ABLATION TYPE
# $3 DATASET
# $4 VERSION
# eg. ./run_main 0 DeepXML EURLex-4K 0
# eg. ./run_main 0 DeepXML-fr EURLex-4K 0

export CUDA_VISIBLE_DEVICES=$1
model_type=$2
dataset=$3

source "../configs/${model_type}/${dataset}.sh"
version=$4

create_splits () {
    # $1: dataset
    # $2: train_feat_fname
    # $3: train_label_fname
    # $4: split thresholds
    # $5: temp model data directory
    echo "Creating data splits.."
    python3 ../tools/run_split_data.py $1 $2 $3 $4 $5
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
    flag=$1
    shift
    echo $6
    if [ "${flag}" == "shortlist" ]
    then
        BETA="0.1 0.15 0.2 0.3 0.4 0.5 0.6"
        ./run_base.sh "evaluate" $1 $2 $3 $model_type $4 $5 $6 "${7} ${BETA}"
    else
        BETA="-1"
        ./run_base.sh "evaluate" $1 $2 $3 $model_type $4 $5 $6 "${7} ${BETA}"
    fi
}

work_dir="$HOME/scratch/XC"
data_dir="${work_dir}/data/${dataset}"
temp_model_data="deep-xml_data"

train_file="${data_dir}/train.txt"
test_file="${data_dir}/test.txt"
trn_ft_file="${data_dir}/trn_X_Xf.txt"
trn_lbl_file="${data_dir}/trn_X_Y.txt"
tst_ft_file="${data_dir}/tst_X_Xf.txt"
tst_lbl_file="${data_dir}/tst_X_Y.txt"
mkdir -p "${data_dir}/${temp_model_data}"

convert() {
    perl ../tools/convert_format.pl $1 $2 $3
    perl ../tools/convert_format.pl $4 $5 $6
}

if [ ! -e "${trn_ft_file}" ]; then
    convert ${train_file} ${trn_ft_file} ${trn_lbl_file} ${test_file} ${tst_ft_file} ${tst_lbl_file}
fi



if [ ! -e "${data_dir}/$temp_model_data/$split_threshold/split_stats.json" ]
then
    mkdir -p "${data_dir}/$temp_model_data/$split_threshold"
    echo "Splitting data."
    create_splits $data_dir "${trn_ft_file}" "${trn_lbl_file}" $split_threshold $temp_model_data
else
    echo "Using old" "${data_dir}/$temp_model_data/$split_threshold/split_stats.json"
fi

run(){
    file=$1
    version=$2
    splitid=$3
    learning_rate=$4
    num_epochs=$5
    dlr_step="dlr_step_${file}"
    num_epochs="num_epochs_${file}"
    batch_size="batch_size_${file}"
    num_centriods="num_centroids_${file}"
    echo "Training $file split.. with lr:" ${learning_rate} "epochs:" ${!num_epochs}  
    args="$dataset $version $splitid $use_post $learning_rate $embedding_dims \
            ${!num_epochs} $dlr_factor ${!dlr_step} ${!batch_size} ${work_dir} \
            $model_type ${temp_model_data} ${split_threshold} ${topk} ${!num_centriods} \
            ${use_ensemble}"
    echo $args
    ./run_"${file}".sh $args
}

run_ensemble1(){
    lr_arr="lr_ensemble"
    run "ensemble" $version "-1" ${!lr_arr}
    mv "$results_dir/test_predictions_ensemble_clf.npz" "$results_dir/test_predictions_ensemble_level=0_clf.npz"
    mv "$results_dir/train_predictions_clf.npz" "$results_dir/train_predictions_ensemble_level=0_clf.npz"
    cp "$results_dir/-1/test_predictions_ensemble_combined.npz" "$results_dir/test_predictions_ensemble_clf.npz"
}

for((lr_idx=0; lr_idx<$learning_rates; lr_idx++));
do 

    results_dir="${work_dir}/results/$model_type/${dataset}/v_${version}"
    models_dir="${work_dir}/models/$model_type/${dataset}/v_${version}"

    if [ $num_splits -eq 0 ]; then
        echo "Not using any split to train."
        type="order[$num_splits]"
        epc_arr="num_epochs_${!type}"
        lr_arr="lr_${!type}[${lr_idx}]"
        run "${!type}" $version "-1" ${!lr_arr} ${!epc_arr}
        cp -r ${results_dir}/"-1"/* ${results_dir}
        if [ $use_post -eq 1 ]
        then
            run_beta "shortlist" $dataset $work_dir $version "predictions" $A $B $evaluation_type
        else
            run_beta ${!type} $dataset $work_dir $version "predictions" $A $B $evaluation_type
        fi

    else
        for((sp_idx=$num_splits; sp_idx>0; sp_idx--));
        do
            arg=$(expr $sp_idx - 1 |bc)
            type="order[$arg]"
            lr_arr="lr_${!type}[${lr_idx}]"
            run "${!type}" $version $arg ${!lr_arr}
        done

        if [ $use_post -eq 1 ]
        then
            merge_split_predictions "${results_dir}" "0,1" "test_predictions_knn.npz" "${data_dir}/$temp_model_data/$split_threshold" $num_labels
            merge_split_predictions "${results_dir}" "0,1" "test_predictions_clf.npz" "${data_dir}/$temp_model_data/$split_threshold" $num_labels
        fi
        echo "Evaluating with A/B: ${A}/${B}" $evaluation_type
        run_beta "shortlist" $dataset $work_dir $version "test_predictions" $A $B $evaluation_type
        if [ $use_ensemble -eq 1 ]
        then
            ln -s "$results_dir/1/test_predictions_clf.npz" "$results_dir/1/test_predictions_ensemble_clf.npz"
            ln -s "$results_dir/1/test_predictions_knn.npz" "$results_dir/1/test_predictions_ensemble_knn.npz"
            merge_split_predictions "${results_dir}" "0,1" "test_predictions_ensemble_clf.npz" "${data_dir}/$temp_model_data/$split_threshold" $num_labels
            merge_split_predictions "${results_dir}" "0,1" "test_predictions_ensemble_knn.npz" "${data_dir}/$temp_model_data/$split_threshold" $num_labels
            run_beta "shortlist" $dataset $work_dir $version "test_predictions_ensemble" $A $B $evaluation_type
            
            merge_split_predictions "${results_dir}" "0,1" "train_predictions_clf.npz" "${data_dir}/$temp_model_data/$split_threshold" $num_labels

            mkdir -p "$models_dir/-1"
            cp "$results_dir/test_predictions_ensemble_clf.npz" "$models_dir/-1/test_shortlist.npz"
            cp "$results_dir/train_predictions_clf.npz" "$models_dir/-1/train_shortlist.npz"
            run_ensemble1
            run_beta "shortlist" $dataset $work_dir $version "test_predictions_ensemble" $A $B $evaluation_type
        fi
    fi
    ((version++))
done       

#clean_up
