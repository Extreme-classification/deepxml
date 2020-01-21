version=0
use_post=1
evaluation_type=1
num_splits=2
split_threshold=6
topk=50
embedding_dims=300
learning_rates=1
dlr_factor=0.5
num_labels=670091
A=0.6
B=2.6
use_reranker=1
ns_method=kcentroid

lr_full=(0.02)
num_epochs_full=25
num_centroids_full=1
batch_size_full=255
dlr_step_full=14


lr_shortlist=(0.002)
num_epochs_shortlist=20
num_centroids_shortlist=1
batch_size_shortlist=255
dlr_step_shortlist=10


lr_reranker=(0.005)
num_epochs_reranker=10
num_centroids_reranker=1
batch_size_reranker=510
dlr_step_reranker=8


order=("shortlist" "full")