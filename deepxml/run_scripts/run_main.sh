#!/bin/bash
# $1 Datset
# $2 Version
# $3 1: to do post processsing else 
# $4 Type of evaluation function to be used
# $5 Number of splits for the dataset labels
# $6 Threshold to split on
# $nargs Version to run on each split

#activate_anaconda
export CUDA_VISIBLE_DEVICES=2,3
# cd $HOME/scratch/lab/xctools
# python setup.py install --user
# exit
create_splits () {
    # $1: dataset
    # $2: train_fname
    # $3: num_splits
    echo "Creating data splits.."
    python3 ../tools/run_split_data_on_freq.py $1 $2 $3
}

merge_split_predictions () {
    # $1: fnames_predictions 
    # $2: fnames_mapping
    # $3: num_labels
    # $4: out_fname 
    echo "Merging predictions.."
    python3 ../tools/merge_split_predictions.py $1 $2 $3 $4
}

run_beta(){
    flag=$1
    shift
    if [ "${flag}" == "shortlist" ]
    then
        BETA="0.1 0.15 0.2 0.3 0.4 0.5 0.6"
        ./run_base.sh "evaluate" $1 $2 $3 $4 $5 $6 "${7} ${BETA}"
    else
        BETA="-1"
        ./run_base.sh "evaluate" $1 $2 $3 $4 $5 $6 "${7} ${BETA}"
    fi
}

work_dir="/mnt/XC"
dataset=$1
version=$2
use_post=$3
evaluation_type=$4
num_splits=$(expr $5 - 1)
split_threshold=$6

shift 6

learning_rates=1
lr_full=(0.03)
lr_shortlist=(0.005)
num_epochs_full=25
num_epochs_shortlist=15

embedding_dims=300
dlr_factor=0.5
dlr_step=20
batch_size=255
stats=`echo $dataset | python3 -c "import sys, json; print(json.load(open('../data_stats.json'))[sys.stdin.readline().rstrip()])"` 
stats=($(echo $stats | tr ',' "\n"))

num_labels=${stats[1]}
A=${stats[3]}
B=${stats[4]}
data_dir="${work_dir}/data/${dataset}"

if [ ! -e "${data_dir}/split_stats.json" ]
then
    echo "Splitting data."
    create_splits $data_dir 'train.txt' $split_threshold
else
    echo "Using old" "${data_dir}/split_stats.json"
fi

for((lr_idx=0; lr_idx<$learning_rates; lr_idx++));
do 

    results_dir="${work_dir}/results/deep-xml/${dataset}/v_${version}"

    if [ $num_splits -lt 0 ]; then
        echo "Not using any split to train."
        arg=$(expr $num_splits + 2 |bc)
        lr_arr="lr_${!arg}[${lr_idx}]"
        epc_arr="num_epochs_${!arg}"
        echo "Training ${!arg} split.. with lr:" ${!lr_arr} "epochs:" ${!epc_arr}
        ./run_"${1}".sh $dataset $version $num_splits $use_post ${!lr_arr} $embedding_dims ${!epc_arr} $dlr_factor $dlr_step $batch_size "${work_dir}"
        cp -r ${results_dir}/$num_splits/* ${results_dir}
        echo ${results_dir}
        if [ $use_post -eq 1 ]
        then
            run_beta "shortlist" $dataset $work_dir $version "predictions" $A $B $evaluation_type
        else
            run_beta $1 $dataset $work_dir $version "predictions" $A $B $evaluation_type
        fi

    #TODO shift is not acceptable here
    else
        echo $(seq 0 $num_splits)
        for((sp_idx=$num_splits; sp_idx>=0; sp_idx--));
        do
            arg=$(expr $sp_idx + 1 |bc)
            lr_arr="lr_${!arg}[${lr_idx}]"
            epc_arr="num_epochs_${!arg}"
            echo "Training ${!arg} split.. with lr:" ${!lr_arr} "epochs:" ${!epc_arr}  
            ./run_"${!arg}".sh $dataset $version $sp_idx $use_post ${!lr_arr} $embedding_dims ${!epc_arr} $dlr_factor $dlr_step $batch_size "${work_dir}"
        done
        if [ $use_post -eq 1 ]
        then
            merge_split_predictions "${results_dir}/0/predictions_knn.npz,${results_dir}/1/predictions_knn.npz" "${data_dir}/labels_split_0.txt,${data_dir}/labels_split_1.txt" $num_labels "${results_dir}/predictions_knn.npz"
            merge_split_predictions "${results_dir}/0/predictions_clf.npz,${results_dir}/1/predictions_clf.npz" "${data_dir}/labels_split_0.txt,${data_dir}/labels_split_1.txt" $num_labels "${results_dir}/predictions_clf.npz"
        fi
        echo "Evaluating with A/B: ${A}/${B}"
        run_beta "shortlist" $dataset $work_dir $version "predictions" $A $B $evaluation_type
    fi
    ((version++))
done       

