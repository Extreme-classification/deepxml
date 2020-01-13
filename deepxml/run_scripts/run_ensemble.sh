dataset=$1
version=$2
thresholds=(15 20 25)

index=0
# Run sequentially
for num in ${!thresholds[@]}; do
    echo "Running learner: ${index}.."
    ./run_one.sh 0 DeepXML $dataset "${version}_${index}" ${thresholds[index]}
    ((index++))
done
