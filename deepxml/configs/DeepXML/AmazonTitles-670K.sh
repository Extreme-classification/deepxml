use_post=0
aux_threshold=65536
topk=200
embedding_dims=300
dlr_factor=0.5
num_labels=670091
A=0.6
B=2.6
use_reranker=1
aux_method=0
ns_method='ensemble'

lr_aux=(0.02)
num_epochs_aux=25
num_centroids_aux=1
batch_size_aux=255
dlr_step_aux=14


lr_org=(0.001)
num_epochs_org=20
num_centroids_org=1
batch_size_org=255
dlr_step_org=10


lr_rnk=(0.002)
num_epochs_rnk=15
num_centroids_rnk=1
batch_size_rnk=255
dlr_step_rnk=10


order=("shortlist" "full")