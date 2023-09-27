# CASSPR
This repository is the official implementation for ICCV2023 paper:

[CASSPR: Cross Attention Single Scan Place Recognition](https://www.robots.ox.ac.uk/~joao/publications/xia_iccv2023.pdf)

## Introduction
CASSPR uses a dual-branch hierarchical cross attention transformer to combine the multi-scale spatial context of voxel-based approaches with the local precision of point-based approaches for single LiDAR scan place recognition.


✅ Up to 62% less inference time

✅ Up to 91% less memory

✅ Excellent loop-closure on KITTI Odometry benchmark

✅ SOTA on USyd, TUM, and Oxford RobotCar


## Citation
```
@InProceedings{Xia_2023_ICCV,
author = {Xia, Yan and Gladkova, Mariia and Wang, Rui and Li, Qianyun and Stilla, Uwe and Henriques, Joao F and Cremers, Daniel},
title = {CASSPR: Cross Attention Single Scan Place Recognition },
booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
month = {October},
year = {2023},
organization={IEEE}
}


@inproceedings{xia2021soe,
title={SOE-Net: A self-attention and orientation encoding network for point cloud based place recognition},
author={Xia, Yan and Xu, Yusheng and Li, Shuang and Wang, Rui and Du, Juan and Cremers, Daniel and Stilla, Uwe},
booktitle={Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition},
pages={11348--11357},
year={2021}
}
```






## Installation
We recommend using conda environment (python 3.8).
```bash
conda create --name casspr python=3.8
conda activate casspr
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install numpy ninja tqdm tensorboard scikit-learn
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps # commit 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
pip install open3d torchtyping linear_attention_transformer future_fstrings bitarray pytorch_metric_learning==1.1.2 psutil


cd models/transformer/cuda_ops
python setup.py install
```

## Dataset preparation
Download the Used, Oxford, and TUM dataset, and then run the following scripts to prepare the data, for example,

```
cd generating_queries/


# Generate training tuples for the USyd Dataset
python generate_training_tuples_usyd.py


# Generate evaluation tuples for the USyd Dataset
python generate_test_sets_usyd.py
```

## Training and evaluation
To train our CASSPR, change ```dataset_folder``` in the config file and run:
```
cd training

# To train the desired model on the USyd Dataset
python train.py --config ../config/config_usyd.txt --model_config ../config/model_config_usyd.txt

```

We release pre-trained models on the USyd and Oxford datasets in https://github.com/Yan-Xia/CASSPR/tree/main/pretrained. Please note that the results are slightly different from the ones reported in the main paper (see [notes](https://github.com/Yan-Xia/CASSPR/blob/main/pretrained/notes.txt)).

To evaluate the pre-trained USyd model, run:
```
cd eval

python evaluate.py --config ../config/config_usyd.txt --model_config ../config/model_config_usyd.txt --weights ../pretrained/CASSPR-USyd.pth
```
## References:
The code is in heavily built on [MinkLoc3D-SI](https://github.com/KamilZywanowski/MinkLoc3D-SI).

1. Uy, M.A. and Lee, G.H., 2018. Pointnetvlad: Deep point cloud based retrieval for large-scale place recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 4470-4479).
2. Xia, Y., Xu, Y., Li, S., Wang, R., Du, J., Cremers, D. and Stilla, U., 2021. SOE-Net: A self-attention and orientation encoding network for point cloud based place recognition. In *Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition* (pp. 11348-11357).
3. Żywanowski, K., Banaszczyk, A., Nowicki, M.R. and Komorowski, J., 2021. MinkLoc3D-SI: 3D LiDAR place recognition with sparse convolutions, spherical coordinates, and intensity. *IEEE Robotics and Automation Letters*, *7*(2), pp.1079-1086.

