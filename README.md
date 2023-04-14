## BARISTA: Barzilai-Borwein Iterative Step Tuning Algorithm

![alt text](logo.png "barista")

## Description
`barista` contains an implementation of an algorithm to automatically compute step sizes for *Stochastic Gradient Descent* and it's variant : *Stochastic Variance Reduced Gradient* using the *Barzilai-Borwein* method. This approach is based on the research paper by Tan *et.al.* presented at NIPS 2016 [[1]](https://proceedings.neurips.cc/paper/2016/hash/c86a7ee3d8ef0b551ed58e354a836f2b-Abstract.html). The project aims to solve two common problems in Machine Learning: Logistic regression and Squared Hinge Loss-based SVM, both with $l_2$ regularization. The algorithm is tested on three different classification datasets obtained from LIBSVM, and the results are compared with those presented in [1]. This is in partial fullfilment for the course E1 260 : Optimization for Machine Learning, January 2023 Term, IISc Bangalore. All the code has been written in [Python 3](https://www.python.org).

## Organisation of the `barista` package
* `./src` contains the following scripts.
    - `01. run.py`
    - `02. sgd.py`
    - `03. libsvm_downloader.py`
    - `04. libsvm_dataset_helper.py`
    - `05. validation_plots.py`

* `./data` contains the three datasets from LIBSVM (download using `libsvm_downloader.py`).

* `./results` contains the plots obtained from `validation_plots.py`.

## Authors
* [Aditya C](mailto:adichand20@gmail.com), Department of Electrical and Communication Engineering, IISc Bangalore.
* [Rajesh Berepalli](mailto:rajeshberepa@iisc.ac.in), Department of Electrical and Communication Engineering, IISc Bangalore.
> *For questions or suggestions, please contact: adichand20@gmail.com or rajeshberepa@iisc.ac.in*

## References
<a id="1">[1]</a> 
Tan, Conghui, Shiqian Ma, Yuhong Dai and Yuqiu Qian. “Barzilai-Borwein Step Size for Stochastic Gradient Descent.” NIPS (2016).
