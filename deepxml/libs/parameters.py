from libs.parameters_base import ParametersBase


class Parameters(ParametersBase):
    """
        Parameter class for XML Classifiers
    """

    def __init__(self, description):
        super().__init__(description)
        self._construct()

    def _construct(self):
        super()._construct()
        self.parser.add_argument(
            '--trans_method',
            dest='trans_method',
            default='non_linear',
            type=str,
            action='store',
            help='which network to use')
        self.parser.add_argument(
            '--lr',
            dest='learning_rate',
            default=0.1,
            action='store',
            type=float,
            help='Learning rate')
        self.parser.add_argument(
            '--dlr_step',
            dest='dlr_step',
            default=7,
            action='store',
            type=int,
            help='dlr_step')
        self.parser.add_argument(
            '--last_saved_epoch',
            dest='last_epoch',
            default=0,
            action='store',
            type=int,
            help='Last saved model at this epoch!')
        self.parser.add_argument(
            '--last_epoch',
            dest='last_epoch',
            default=0,
            action='store',
            type=int,
            help='Start training from here')
        self.parser.add_argument(
            '--ann_method',
            dest='ann_method',
            default='hnsw',
            action='store',
            type=str,
            help='Approximate nearest neighbor method')
        self.parser.add_argument(
            '--ann_threads',
            dest='ann_threads',
            default=4,
            action='store',
            type=int,
            help='HSNW params')
        self.parser.add_argument(
            '--num_hashes',
            dest='num_hashes',
            default=-1,
            action='store',
            type=int,
            help='#Hash functions to use')
        self.parser.add_argument(
            '--num_buckets',
            dest='num_buckets',
            default=-1,
            action='store',
            type=int,
            help='#buckets to hash vocabulary')
        self.parser.add_argument(
            '--label_indices',
            dest='label_indices',
            default=None,
            action='store',
            type=str,
            help='Use these labels only')
        self.parser.add_argument(
            '--feature_indices',
            dest='feature_indices',
            default=None,
            action='store',
            type=str,
            help='Use these features only')
        self.parser.add_argument(
            '--efC',
            dest='efC',
            action='store',
            default=300,
            type=int,
            help='efC')
        self.parser.add_argument(
            '--num_nbrs',
            dest='num_nbrs',
            action='store',
            default=300,
            type=int,
            help='num_nbrs')
        self.parser.add_argument(
            '--efS',
            dest='efS',
            action='store',
            default=300,
            type=int,
            help='efS')
        self.parser.add_argument(
            '--M',
            dest='M',
            action='store',
            default=100,
            type=int,
            help='M')
        self.parser.add_argument(
            '--retrain_hnsw_after',
            action='store',
            default=1,
            type=int,
            help='Retrain HSNW after these many epochs!')
        self.parser.add_argument(
            '--num_labels',
            dest='num_labels',
            default=-1,
            action='store',
            type=int,
            help='#labels')
        self.parser.add_argument(
            '--vocabulary_dims',
            dest='vocabulary_dims',
            default=-1,
            action='store',
            type=int,
            help='#features')
        self.parser.add_argument(
            '--padding_idx',
            dest='padding_idx',
            default=0,
            action='store',
            type=int,
            help='padding_idx')
        self.parser.add_argument(
            '--out_fname',
            dest='out_fname',
            default='out',
            action='store',
            type=str,
            help='prediction file name')
        self.parser.add_argument(
            '--dlr_factor',
            dest='dlr_factor',
            default=0.5,
            action='store',
            type=float,
            help='dlr_factor')
        self.parser.add_argument(
            '--m',
            dest='momentum',
            default=0.9,
            action='store',
            type=float,
            help='momentum')
        self.parser.add_argument(
            '--w',
            dest='weight_decay',
            default=0.0,
            action='store',
            type=float,
            help='weight decay parameter')
        self.parser.add_argument(
            '--dropout',
            dest='dropout',
            default=0.5,
            action='store',
            type=float,
            help='Dropout')
        self.parser.add_argument(
            '--optim',
            dest='optim',
            default='SGD',
            action='store',
            type=str,
            help='Optimizer')
        self.parser.add_argument(
            '--embedding_dims',
            dest='embedding_dims',
            default=300,
            action='store',
            type=int,
            help='embedding dimensions')
        self.parser.add_argument(
            '--embeddings',
            dest='embeddings',
            default='fasttextB_embeddings_300d.npy',
            action='store',
            type=str,
            help='embedding file name')
        self.parser.add_argument(
            '--hidden_dims',
            dest='hidden_dims',
            default=300,
            action='store',
            type=int,
            help='units in penultimate layer')
        self.parser.add_argument(
            '--num_epochs',
            dest='num_epochs',
            default=20,
            action='store',
            type=int,
            help='num epochs')
        self.parser.add_argument(
            '--batch_size',
            dest='batch_size',
            default=64,
            action='store',
            type=int,
            help='batch size')
        self.parser.add_argument(
            '--num_centroids',
            dest='num_centroids',
            default=1,
            type=int,
            action='store',
            help='#Centroids (Use multiple for ext head if more than 1)')
        self.parser.add_argument(
            '--low_rank',
            dest='low_rank',
            default=-1,
            type=int,
            action='store',
            help='#dim of low dimensional space')
        self.parser.add_argument(
            '--beta',
            dest='beta',
            default=0.2,
            type=float,
            action='store',
            help='weight of classifier')
        self.parser.add_argument(
            '--res_init',
            dest='res_init',
            default='eye',
            type=str,
            action='store',
            help='eye or random')
        self.parser.add_argument(
            '--label_padding_index',
            dest='label_padding_index',
            default=None,
            type=int,
            action='store',
            help='Pad with this')
        self.parser.add_argument(
            '--mode',
            dest='mode',
            default='train',
            type=str,
            action='store',
            help='train or predict')
        self.parser.add_argument(
            '--keep_invalid',
            action='store_true',
            help='Keep labels which do not have any training instance!.')
        self.parser.add_argument(
            '--freeze_embeddings',
            action='store_true',
            help='Do not train word embeddings.')
        self.parser.add_argument(
            '--use_residual',
            action='store_true',
            help='Use residual connection')
        self.parser.add_argument(
            '--use_low_rank',
            action='store_true',
            help='Use low rank on classifier')
        self.parser.add_argument(
            '--use_shortlist',
            action='store_true',
            help='Use shortlist or full')
        self.parser.add_argument(
            '--use_head_embeddings',
            action='store_true',
            help='Use embeddings from head or default')
        self.parser.add_argument(
            '--validate',
            action='store_true',
            help='Validate or just train')
        self.parser.add_argument(
            '--shuffle',
            action='store',
            default=True,
            type=bool,
            help='Shuffle data during training!')
        self.parser.add_argument(
            '--device_embeddings',
            action='store',
            default='cuda:0',
            help='Device for embeddings'
        )
        self.parser.add_argument(
            '--device_classifier',
            action='store',
            default='cuda:0',
            help='Device for classifier'
        )
        self.parser.add_argument(
            '--use_hash_embeddings',
            action='store_true',
            help='Use embeddings'
        )
        self.parser.add_argument(
            '--normalize',
            action='store_true',
            help='Normalize features or not!')
        self.parser.add_argument(
            '--use_sparse',
            action='store_true',
            help='Use Sparse Features')
        self.parser.add_argument(
            '--update_shortlist',
            action='store_true',
            help='Update shortlist while predicting'
        )
