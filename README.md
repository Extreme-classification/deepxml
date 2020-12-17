# DeepXML

Code for _DeepXML: A Deep Extreme Multi-Label Learning Framework Applied to Short Text Documents_

## Requirements

---

* Pyxclib
* NumPy
* PyTorch
* Numba
* Scikit-learn

---

## Example use cases

---

### DeepXML framework

The Astec algorithm can be trained using DeepXML framework as follows.

```bash
./run_main.sh 0 DeepXML Astec EURLex-4K 0 22
```

### DeepXML-OVA framework

The Astec algorithm can be trained in a One-vs.-All fashion as follows.

```bash
./run_main.sh 0 DeepXML-OVA Astec EURLex-4K 0 22
```

### DeepXML-ANNS framework

The Astec algorithm can be trained with an Approximate nearest neighbours structure (ANNS) trained on label centroids. Support is available for a fixed graph or periodic training of the ANNS graph.

```bash
./run_main.sh 0 DeepXML-ANNS Astec EURLex-4K 0 22
```

## Full documentation

---

```txt
Expected directory structure
+-- work_dir
|  +-- programs
|  |  +-- deepxml
|  |    +-- deepxml
|  +-- data_dir
|    +-- dataset
|  +-- model_dir
|  +-- results_dir

```

```txt
./run_main.sh <gpu_id> <framework> <method> <dataset> <version> <seed>

* gpu_id: Run the program on this GPU.

* framework
  - DeepXML: Divides the XML problems in 4 modules as proposed in the paper.
  - DeepXML-OVA: Train the method in one-vs.-all fashion [4][5].
  - DeepXML-ANNS: Train the method using a label shortlist.

* The code uses a json file to construct the feature architecture. Features could be computed using following methods
  - Astec: As proposed in the DeepXML paper [1]
  - RNN: RNN based sequential models
  - XML-CNN: CNN architecture as proposed in the XML-CNN paper [4].

* dataset
  - Name of the dataset.
  - Expected files in work_dir/data/<dataset>
    - train.txt
    - test.txt

* version
  - different runs could be managed by versions 
  - models and results are stored with this argument

* seed
  - seed value as used by numpy and PyTorch
```

## References

---
[1] DeepXML

[2] pyxclib: <https://github.com/kunaldahiya/pyxclib>

[3] H. Jain,  V. Balasubramanian,  B. Chunduri and M. Varma, Slice: Scalable linear extreme classifiers trained on 100 million labels for related searches, in WSDM 2019.

[4] J. Liu,  W.-C. Chang,  Y. Wu and Y. Yang, Slice: Deep Learning for Extreme Multi-label Text Classification, in SIGIR 2017.

[5]  R. Babbar, and B. Schölkopf, DiSMEC - Distributed Sparse Machines for Extreme Multi-label Classification in WSDM, 2017.
