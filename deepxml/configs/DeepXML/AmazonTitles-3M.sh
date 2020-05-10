version=0
use_post=1
evaluation_type=1
num_splits=2
split_threshold=75
topk=50
embedding_dims=300
learning_rates=1
dlr_factor=0.5
num_labels=2812281
A=0.6
B=2.6
use_reranker=0
ns_method=kcentroid

lr_aux=(0.005)
num_epochs_aux=25
num_centroids_aux=1
batch_size_aux=255
dlr_step_aux=14


lr_org=(0.002)
num_epochs_org=20
num_centroids_org=300
batch_size_org=255
dlr_step_org=10


lr_rnk=(0.002)
num_epochs_rnk=15
num_centroids_rnk=1
batch_size_rnk=255
dlr_step_rnk=8


order=("shortlist" "full")
