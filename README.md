# CASSPR
The codes for ICCV2023 paper ' CASSPR: Cross Attention Single Scan Place Recognition'

## Installation

We recommend using conda environment (python 3.8).

```bash
conda create --name casspr python=3.8
conda activate casspr
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install numpy ninja tqdm tensorboard scikit-learn
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps # commit 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
pip install open3d torchtyping linear_attention_transformer future_fstrings bitarray pytorch_metric_learning

cd models/cuda_ops
python setup.py install
```