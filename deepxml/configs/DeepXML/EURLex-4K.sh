version=10
use_post=1
evaluation_type=1
num_splits=2
split_threshold=5

learning_rates=1
lr_full=(0.02)
lr_shortlist=(0.003)
lr_ns=(0.003)
num_epochs_full=1
num_epochs_shortlist=1

embedding_dims=300
dlr_factor=0.5
dlr_step=14
batch_size=128

num_labels=3993
A=0.55
B=1.5
order=("shortlist" "full")