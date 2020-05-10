version=0
use_post=0
evaluation_type=1
num_splits=2
aux_threshold=65536
topk=100
embedding_dims=300
learning_rates=1
dlr_factor=0.5
num_labels=352072
A=0.55
B=1.5
aux_method=0
ns_method='kcentroid'

lr_aux=(0.005)
num_epochs_aux=25
num_centroids_aux=1
batch_size_aux=255
dlr_step_aux=14


lr_org=(0.002)
num_epochs_org=20
num_centroids_org=1
batch_size_org=255
dlr_step_org=10


lr_reranker=(0.002)
num_epochs_reranker=15
num_centroids_reranker=1
batch_size_reranker=255
dlr_step_reranker=8

order=("shortlist" "full")