dataset=$1
version=$2
seeds=(22 33 666)

index=0
# Run sequentially
for seed in ${!seeds[@]}; do
    echo "Running learner: ${index}.."
    ./run_main.sh 0 DeepXML $dataset "${version}_${index}" ${seed}
    ((index++))
done
