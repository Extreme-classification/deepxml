version=0
use_post=0
evaluation_type=1
num_splits=2
split_threshold=65536
topk=500
embedding_dims=300
learning_rates=1
dlr_factor=0.5
num_labels=501070
A=0.5
B=0.4
aux_method=0
ns_method='kcentroid'
use_reranker=1

lr_aux=(0.005)
num_epochs_aux=25
num_centroids_aux=300
batch_size_aux=255
dlr_step_aux=14


lr_org=(0.002)
num_epochs_org=20
num_centroids_org=1
batch_size_org=255
dlr_step_org=10


lr_rnk=(0.002)
num_epochs_rnk=15
num_centroids_rnk=1
batch_size_rnk=255
dlr_step_rnk=8

order=("shortlist" "full")