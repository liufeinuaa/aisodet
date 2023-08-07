# AISODET

The offical implementation of the ESRTMDet model on the IEEE JSTARS paper “ESRTMDet: An End-to-End Super-Resolution
Enhanced Real-Time Rotated Object Detector
for Degraded Aerial Images”

## Introduction
### aisodet工程的架构
- aisodet/ 主要包括自定义的一些数据集的实现，和往mmengine中进行模型注册的注册器和一些常用的于mmengine进行交互的工具的实现
- configs/ 是一些只依赖于mmdet 和 mmrotate 模型运行测试时的 cfg
- projects/ 为开发的模型(esrtmdet)和具体用到的方法
- tools/ 为调用mmenigne进行训练和测试的接口，及一些数据集处理函数的方法

### 依赖的开源库
```
torch                         1.9.0+cu111  #torch 1.13.1 也没问题
torchvision                   0.10.0+cu111 
# openmmlab的库
mmengine                      0.7.2
mmcv                          2.0.0
mmdet                         3.0.0 
mmrotate                      1.0.0rc1
```
自行参考openmmlab官方库中的安装方法，安装对应的依赖库

### 其他注意事项
数据集及训练权重文件夹
```
# 训练log及权重文件等的路径的软链接
ln -s /media/liufei/ubdata/work_dirs/ work_dirs # 可将你的文件路径替换进去
# 数据集文件路径的软链接
ln -s /media/liufei/ubm2/datasets/ data2 # 可将你的文件路径替换进去
```
数据预处理
```
# 自行查看 tools/data/dota 和 tools/sr 中的 readme.md
```

## Installation
```
python setup.py develop
```


## Models
DOTA 1.0 所有尺寸模型的权重 链接: https://pan.baidu.com/s/12On2eJ2ngk30RAcKSXLKGg 提取码: 9zrv 

UCAS-AOD 所有尺寸模型的权重 链接: https://pan.baidu.com/s/10Gt-p8xOmKGjA-rFWyOcPA 提取码: jbd9 


## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```bibtex
@ARTICLE{10130374,
  author={Liu, Fei and Chen, Renwen and Zhang, Junyi and Ding, Shanshan and Liu, Hao and Ma, Shaofei and Xing, Kailing},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={ESRTMDet: An End-to-End Super-Resolution Enhanced Real-Time Rotated Object Detector for Degraded Aerial Images}, 
  year={2023},
  volume={16},
  number={},
  pages={4983-4998},
  doi={10.1109/JSTARS.2023.3278295}
}
```

## License
This project is released under the [Apache 2.0 license](LICENSE).
