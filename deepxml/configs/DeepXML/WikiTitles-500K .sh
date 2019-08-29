version=0
use_post=1
evaluation_type=1
num_splits=2
split_threshold=25

learning_rates=1
lr_full=(0.005)
lr_shortlist=(0.002)

num_epochs_full=25
num_epochs_shortlist=20

embedding_dims=300
dlr_factor=0.5
dlr_step_full=14
dlr_step_shortlist=10

batch_size_full=255
batch_size_shortlist=255

num_labels=501070
A=0.5
B=0.4
order=("shortlist" "full")