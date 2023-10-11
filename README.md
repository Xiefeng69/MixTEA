# MixTEA

## Getting Started

### Datasets
We use entity alignment benchmark datasets **OpenEA** which can be downloaded from [OpenEA](https://github.com/nju-websoft/OpenEA). You need to put the prepared data into `/data` folder.

### Dependencies
+ Python 3
+ PyTorch
+ networkx==2.5.1
+ Scipy
+ Numpy
+ Pandas
+ Scikit-learn
+ Faiss

You can automatically download corresponding dependencies by following scripts:
```
pip install -r .\requirements.txt
```

### Running
Note: The settings of hyper-parameters are given in `/args` folder.

To run MixTEA, please use the following scripts (ps: --task is an argument):
```
python run.py --task en_fr_15k
python run.py --task en_de_15k
python run.py --task d_w_15k
python run.py --task d_y_15k
```

To run 5-fold cross-validation, please use the following script:
```
python run_fold.py --task en_fr_15k
```

We also provide jupyter notebook version in `MixTEA.ipynb`.

> If you have any difficulty or question in running code and reproducing experimental results, please email to xiefeng@nudt.edu.cn.

## Acknowledgement
We refer to the codes of these repos: GCN-Align, OpenEA, MuGNN, MeanTeacher. Thanks for their great contributions!