# DeepXML

Code for _DeepXML: A Deep Extreme Multi-Label Learning Framework Applied to Short Text Documents_

---

## Architectures and algorithms

DeepXML supports multiple feature architectures such as Bag-of-embedding/Astec, RNN, CNN etc. The code uses a json file to construct the feature architecture. Features could be computed using following encoders:

* Bag-of-embedding/Astec: As used in the DeepXML paper [1].
* RNN: RNN based sequential models. Support for RNN, GRU, and LSTM.
* XML-CNN: CNN architecture as proposed in the XML-CNN paper [4].

---

## Best Practices for features creation

---

* Adding sub-words on top of unigrams to the vocabulary can help in training more accurate embeddings and classifiers.

---

## Setting up

---

### Expected directory structure

```txt
+-- <work_dir>
|  +-- programs
|  |  +-- deepxml
|  |    +-- deepxml
|  +-- data
|    +-- <dataset>
|  +-- models
|  +-- results

```

### Download data for Astec

```txt
* Download the (zipped file) BoW features from XML repository.  
* Extract the zipped file into data directory. 
* The following files should be available in <work_dir>/data/<dataset> for new datasets (ignore the next step)
    - trn_X_Xf.txt
    - trn_X_Y.txt
    - tst_X_Xf.txt
    - tst_X_Y.txt
    - fasttextB_embeddings_300d.npy or fasttextB_embeddings_512d.npy
* The following files should be available in <work_dir>/data/<dataset> if the dataset is in old format (please refer to next step to convert the data to new format)
    - train.txt
    - test.txt
    - fasttextB_embeddings_300d.npy or fasttextB_embeddings_512d.npy 
```

### Convert to new data format

```perl
# A perl script is provided (in deepxml/tools) to convert the data into new format as expected by Astec
# Either set the $data_dir variable to the data directory of a particular dataset or replace it with the path
perl convert_format.pl $data_dir/train.txt $data_dir/trn_X_Xf.txt $data_dir/trn_X_Y.txt
perl convert_format.pl $data_dir/test.txt $data_dir/tst_X_Xf.txt $data_dir/tst_X_Y.txt
```

## Example use cases

---

### A single learner with DeepXML framework

The DeepXML framework can be utilized as follows. A json file is used to specify architecture and other arguments. Please refer to the full documentation below for more details.

```bash
./run_main.sh 0 DeepXML EURLex-4K 0 108
```

### An ensemble of multiple learners with DeepXML framework

An ensemble can be trained as follows. A json file is used to specify architecture and other arguments.

```bash
./run_main.sh 0 DeepXML EURLex-4K 0 108,666,786
```

## Full Documentation

```txt
./run_main.sh <gpu_id> <framework> <dataset> <version> <seed>

* gpu_id: Run the program on this GPU.

* framework
  - DeepXML: Divides the XML problems in 4 modules as proposed in the paper.
  - DeepXML-OVA: Train the architecture in 1-vs-all fashion [4][5], i.e., loss is computed for each label in each iteration.
  - DeepXML-ANNS: Train the architecture using a label shortlist. Support is available for a fixed graph or periodic training of the ANNS graph.

* dataset
  - Name of the dataset.
  - Astec expects the following files in <work_dir>/data/<dataset>
    - trn_X_Xf.txt
    - trn_X_Y.txt
    - tst_X_Xf.txt
    - tst_X_Y.txt
    - fasttextB_embeddings_300d.npy or fasttextB_embeddings_512d.npy
  - You can set the 'embedding_dims' in config file to switch between 300d and 512d embeddings.

* version
  - different runs could be managed by version and seed.
  - models and results are stored with this argument.

* seed
  - seed value as used by numpy and PyTorch.
  - an ensemble is learned if multiple comma separated values are passed.
```

### Notes

```txt
* Other file formats such as npy, npz, pickle are also supported.
* Initializing with token embeddings (computed from FastText) leads to noticible accuracy gain in Astec. Please ensure that the token embedding file is available in data directory, if 'init=token_embeddings', otherwise it'll throw an error.
* Config files are made available in deepxml/configs/<framework>/<method> for datasets in XC repository. You can use them when trying out Astec/DeepXML on new datasets.
* We conducted our experiments on a 24-core Intel Xeon 2.6 GHz machine with 440GB RAM with a single Nvidia P40 GPU. 128GB memory should suffice for most datasets.
* Astec make use of CPU (mainly for nmslib) as well as GPU. 
```

## Cite as

```bib
@InProceedings{Dahiya21,
    author = "Dahiya, K. and Saini, D. and Mittal, A. and Shaw, A. and Dave, K. and Soni, A. and Jain, H. and Agarwal, S. and Varma, M.",
    title = "DeepXML: A Deep Extreme Multi-Label Learning Framework Applied to Short Text Documents",
    booktitle = "Proceedings of the ACM International Conference on Web Search and Data Mining",
    month = "March",
    year = "2021"
}
```


## YOU MAY ALSO LIKE
- [DECAF: Deep Extreme Classification with Label Features](https://github.com/Extreme-classification/DECAF)
- [GalaXC: Graph Neural Networks with Labelwise Attention for Extreme Classification](https://github.com/Extreme-classification/GalaXC)
- [ECLARE: Extreme Classification with Label Graph Correlations](https://github.com/Extreme-classification/ECLARE)

## References

---
[1] K. Dahiya, D. Saini, A. Mittal, A. Shaw, K. Dave, A. Soni, H. Jain, S. Agarwal, and M. Varma. Deepxml:  A deep extreme multi-label learning framework applied to short text documents. In WSDM, 2021.

[2] pyxclib: <https://github.com/kunaldahiya/pyxclib>

[3] H. Jain,  V. Balasubramanian,  B. Chunduri and M. Varma, Slice: Scalable linear extreme classifiers trained on 100 million labels for related searches, In WSDM 2019.

[4] J. Liu,  W.-C. Chang,  Y. Wu and Y. Yang, XML-CNN: Deep Learning for Extreme Multi-label Text Classification, In SIGIR 2017.

[5]  R. Babbar, and B. Schölkopf, DiSMEC - Distributed Sparse Machines for Extreme Multi-label Classification In WSDM, 2017.

[6] P., Bojanowski, E. Grave, A. Joulin, and T. Mikolov. Enriching word vectors with subword information. In TACL, 2017.
