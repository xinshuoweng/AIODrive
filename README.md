# AIODrive

<b>All-In-One Drive: A Large-Scale Comprehensive Perception Dataset with High-Density Long-Range Point Clouds</b>

This repository contains the official implementation for our manuscript "[All-In-One Drive: A Large-Scale Comprehensive Perception Dataset with High-Density Long-Range Point Clouds](https://www.xinshuoweng.com/papers/AIODrive/arXiv.pdf)". Our project website is [here](http://www.aiodrive.org/). If you find our paper or code useful, please cite our paper below:

```
@article{Weng2020_AIODrive,
author = {Weng, Xinshuo and Man, Yunze and Cheng, Dazhi and Park, Jinhyung and O'Toole, 
Matthew and Kitani, Kris},
journal = {arXiv},
title = {{All-In-One Drive: A Large-Scale Comprehensive Perception Dataset with 
High-Density Long-Range Point Clouds}},
year = {2020}
}
```

<img align="center" width="98%" src="https://github.com/xinshuoweng/AIODrive/blob/master/demo.gif">

## Introduction
Developing datasets that cover comprehensive sensors, annotations and full data distribution is important for innovating robust multi-sensor multi-task perception systems. Though many datasets have been released, they target for different use-cases such as 3D segmentation (SemanticKITTI), radar sensing (nuScenes), large-scale training (Waymo). As a result, we are still in need of a dataset that forms a union of various strengths of existing datasets. To address this challenge, we present the AIODrive dataset, a synthetic large-scale dataset that provides comprehensive sensors, annotations and environmental variations. Specifically, we provide (1) eight sensor modalities (RGB, Stereo, Depth, LiDAR, SPAD-LiDAR, Radar, IMU, GPS), (2) annotations for all mainstream perception tasks (\emph{e.g.}, detection, tracking, trajectory prediction, segmentation, depth estimation), and (3) rare driving scenarios such as adverse weather and lighting, crowded scenes, high-speed driving, violation of traffic rules, and accidents. In addition to comprehensive data, long-range perception is also important to perception systems as early detection of faraway objects can help prevent collision in high-speed driving scenarios. However, due to the sparsity and limited range of point cloud data in prior datasets, developing and evaluating long-range perception algorithms is challenging. To address the issue, we provide high-density long-range point clouds for LiDAR and SPAD-LiDAR sensors, about 10$\times$ denser and larger sensing range than Velodyne-64. 

## Dependencies:
This code depends on my personal toolbox: https://github.com/xinshuoweng/Xinshuo_PyToolbox. Please install the toolbox by

*1. Clone the github repository.*
~~~shell
git clone https://github.com/xinshuoweng/Xinshuo_PyToolbox
~~~

*2. Install dependency for the toolbox.*
~~~shell
cd Xinshuo_PyToolbox
pip install -r requirements.txt
~~~
