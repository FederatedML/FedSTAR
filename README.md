# FedSTAR: Federated Self-Training for Semi-Supervised Audio Recognition

Federated Learning is a distributed machine learning paradigm dealing with decentralized and personal datasets. Since data reside on devices like smartphones and virtual assistants, labeling is entrusted to the clients, or labels are extracted in an automated way. Specifically, in the case of audio data, acquiring semantic annotations can be prohibitively expensive and time-consuming. As a result, an abundance of audio data remains unlabeled and unexploited on users' devices. Most existing federated learning approaches focus on supervised learning without harnessing the unlabeled data. In this work, we study the problem of semi-supervised learning of audio models via self-training in conjunction with federated learning. We propose FedSTAR to exploit large-scale on-device unlabeled data to improve the generalization of audio recognition models. We further demonstrate that self-supervised pre-trained models can accelerate the training of on-device models, significantly improving convergence to within fewer training rounds. We conduct experiments on diverse public audio classification datasets and investigate the performance of our models under varying percentages of labeled and unlabeled data. Notably, we show that with as little as 3% labeled data available, FedSTAR on average can improve the recognition rate by 13.28% compared to the fully supervised federated model.

<img src=./images/fedstar.png width=50%/>

A complete description of our work can be found in [our recent ACM publication](https://arxiv.org/abs/2107.06877).

## Dependencies
* Python 3.6+
* [TensorFlow 2.3.1](https://www.tensorflow.org/)
* [TensorFlow Datasets](https://www.tensorflow.org/datasets/overview)

## Dataset Preparation
To prepare the datasets, please follow the instruction given in [here](data_splits/README.md).

## Executing experiments
From the root directory of this repo, run:

```console
foo@bar:~$ ./run.sh
```
You can configure all federated parameters (i.e. number of federated rounds, number of clients, percentage of labelled data, etc.,) from the [config](config.yml) file.

## Reference
If you use this repository, please consider citing:

<pre>article{10.1145/3520128,
	author = {Tsouvalas, Vasileios and Saeed, Aaqib and Ozcelebi, Tanir},
	title = {Federated Self-Training for Semi-Supervised Audio Recognition},
	year = {2022},
	publisher = {Association for Computing Machinery},
	address = {New York, NY, USA},
	issn = {1539-9087},
	url = {https://doi.org/10.1145/3520128},
	doi = {10.1145/3520128},
	journal = {ACM Trans. Embed. Comput. Syst.},
	month = {feb},
}</pre>


<pre>@INPROCEEDINGS{9746356,
  author={Tsouvalas, Vasileios and Saeed, Aaqib and Ozcelebi, Tanir},
  booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Federated Self-Training for Data-Efficient Audio Recognition}, 
  year={2022},
  doi={10.1109/ICASSP43922.2022.9746356}}</pre>
