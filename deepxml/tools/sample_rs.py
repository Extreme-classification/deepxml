import numpy as np
import sys
import _pickle as pickle
import os
import xctools.data.data_utils as du
# np.random.seed(42)
data_dir = './'
train_data = pickle.load(open(os.path.join(data_dir, 'train.pkl'), 'rb'))
test_data = pickle.load(open(os.path.join(data_dir, 'test.pkl'), 'rb'))

tr_feat, tr_labels = train_data['features'], train_data['labels']
ts_feat, ts_labels = test_data['features'], test_data['labels']

freq = np.array(tr_labels.sum(axis=0)).ravel()
def inst_lb(lbs,labels):                                                                
    sampled_instances = np.where(labels[:, lbs].sum(axis=1).ravel()>0)[1]
    selected_labels = np.where(labels[sampled_instances, :].sum(axis=0).ravel()>0)[1]
    print(sampled_instances.shape,selected_labels.shape)
    return sampled_instances, selected_labels

def inst(lbs,labels):
    sampled_instances = np.where(labels[:, lbs].sum(axis=1).ravel()>0)[1]
    print(sampled_instances.shape)
    return sampled_instances

lbs = np.random.choice(tr_labels.shape[1], size=50, replace=False, p=freq**0.65/np.sum(freq**0.65))
_,lbs = inst_lb(lbs, tr_labels)
_,lbs = inst_lb(lbs, tr_labels)
tr_insts,_ = inst_lb(lbs, tr_labels)
ts_insts = inst(lbs,ts_labels)

print("Formatting data")
tr_labels = tr_labels[:,lbs][tr_insts,:]
ts_labels = ts_labels[:,lbs][ts_insts,:]

tr_feat = tr_feat[tr_insts,:]
valid_ft = np.where(tr_feat.sum(axis=0)>0)[1]
tr_feat = tr_feat[:,valid_ft]
ts_feat = ts_feat[ts_insts,:][:,valid_ft]
valid_ts_instances = np.where(ts_feat.sum(axis=1)>0)[0]
ts_feat = ts_feat[valid_ts_instances]
ts_labels = ts_labels[valid_ts_instances]
print(np.where(ts_labels.sum(axis=0)>0)[1].shape)

pickle.dump({"valid_ft":valid_ft,"valid_lb":lbs,"tr_insts":tr_insts,"ts_insts":valid_ts_instances},open('new_rs_data_sampled_stats.pkl','wb'))
emb = np.save('new_cdssm_embeddings_64d.npy',np.load('cdssm_embeddings_64d.npy')[valid_ft,:])
du.write_data('new_train.txt',tr_feat,tr_labels)
du.write_data('new_test.txt',ts_feat,ts_labels)