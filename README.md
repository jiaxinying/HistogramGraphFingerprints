# HistogramGraphFingerprints

This repo provides the source code for the following work.

A Simple Yet Effective Method Improving Graph Fingerprints for Graph-Level Prediction.

Jiaxin Ying\*, Jiaqi Ma\*, Qiaozhu Mei.

## Requirements

Most dependency packages can be installed with `environment.yml` using the following command:

```shell
conda env create -f environment.yml
```

This command will create a conda environment named `HGF`. Use `conda activate HGF` to activate the environment.

This code also has depedency on [PyTorch-Geometric](https://github.com/rusty1s/pytorch_geometric) and [k-gnn](https://github.com/chrsmrrs/k-gnn), which must be installed mannually. To make sure the PyTorch version aligns with the PyTorch-Geometric version, we suggest installing PyTorch using the following command:

```shell
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

For PyTorch-Geometric, please refer to the [official website](https://github.com/rusty1s/pytorch_geometric) for installation instruction. The version we tested are listed below.
- torch-cluster==1.5.8
- torch-geometric==1.6.3
- torch-scatter==2.0.5
- torch-sparse==0.6.8
- torch-spline-conv==1.2.0

For k-gnn, we have included a copy in this repo an it can be installe with the following commands:

```shell
cd k-gnn
pip install .
```

*Note: this code is tested on a machine with CUDA 11.0.*


## Run the Code

Example commands to run the experiments are provided in `test.sh`.



