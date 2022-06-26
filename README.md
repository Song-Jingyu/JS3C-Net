# JS3C-Net

## This is a forked version of JS3C-Net for training on CarlaSC dataset. You can check the results on our [paper](https://arxiv.org/abs/2203.07060)

## Getting started with JS3C-Net on the [CarlaSC](https://umich-curly.github.io/CarlaSC.github.io/) dataset
You can check the information about the data and instructions on downloading on our CarlaSC dataset [website](https://umich-curly.github.io/CarlaSC.github.io/). You can also check our models to do scene completion on the [3DMapping](https://github.com/UMich-CURLY/3DMapping) repo.

### Dependencies
The dependencies is the same as mentioned in JS3C-Net [repo](https://github.com/yanx27/JS3C-Net). We tried our very best to accmodate it to newer version of CUDA toolkit, pytorch and so on but failed. So we released the docker image we used to run the JS3C-Net. The docker image can be downloaded on the [drive](https://drive.google.com/file/d/1swSYThDgAKbzxbl8h5rYROeVFCma3Asw/view?usp=sharing). Everything needed for running JS3C-Net is already installed and the repo can be found in `/home`. The docker command to obatin a container from this image is provided in `docker_command.bash`.
<!-- However, we slightly modify the `spconv` library to suit it to modern GPU architecture. The specific libraries we use are reported below.
 - CUDA and cuDNN
   - We are using CUDA 11.3 and cuDNN 8. The Pytorch version is the latest 
   - We directly compile the operators developed by JS3C-Net by `sh compile.sh` in `/lib`.
   - As to spconv, we tried to use the latest version of official spconv `v2` or `v1.2.1` but failed. So we strongly recommend to use the provided `spconv` in `/lib`. We made a slight change on the provided `spconv` based on this [git issue](https://github.com/pytorch/extension-script/issues/6) to suit it to newer Pytorch. -->
  

### Training
 - To start with using [CarlaSC](https://umich-curly.github.io/CarlaSC.github.io/) dataset, make sure you download the dataset and unzip it.
 - We provided a script `train_js3c_carla.py` for using JS3C-Net.
   - We provide two types of setting files. The default is the reduced-label setting, you can also enabled all-label setting by change the config file path to `carla_all.yaml` in `/opt`. You can get more information about the two different settings in our [paper](paper_link).
   - Change the `data_dir` variable in the notebook. We use a `TODO` comment to make it stand out.
   - Change the `TEST` flag to `False` in the notebook. We use a `TODO` comment to make it stand out.
   - There will be a folder containing the training log, weights etc. in `/Runs` folder.

### Testing
 - We include the testing function in script `test_js3c_carla.py`
   - Change the `MODEL_DIR` variable in the notebook to load the specific weight. We use a `TODO` comment to make it stand out.

## Our SC models
You can check our **MotionSC** model and other implementations of SOTA SC models on the [3DMapping](https://github.com/UMich-CURLY/3DMapping) repo.

---

<br />
<br />


### Sparse Single Sweep LiDAR Point Cloud Segmentation via Learning Contextual Shape Priors from Scene Completion (AAAI2021)
 
This repository is for **JS3C-Net** introduced in the following **AAAI-2021** paper [[arxiv paper]](https://arxiv.org/abs/2012.03762)

Xu Yan, Jiantao Gao, Jie Li, Ruimao Zhang, [Zhen Li*](https://mypage.cuhk.edu.cn/academics/lizhen/), Rui Huang and Shuguang Cui, "Sparse Single Sweep LiDAR Point Cloud Segmentation via Learning Contextual Shape Priors from Scene Completion".

* Semantic Segmentation and Semantic Scene Completion:
![](figure/results.gif)

If you find our work useful in your research, please consider citing:
```
@inproceedings{yan2021sparse,
  title={Sparse Single Sweep LiDAR Point Cloud Segmentation via Learning Contextual Shape Priors from Scene Completion},
  author={Yan, Xu and Gao, Jiantao and Li, Jie and Zhang, Ruimao and Li, Zhen and Huang, Rui and Cui, Shuguang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={4},
  pages={3101--3109},
  year={2021}
}
```

## Getting Started

### Set up
Clone the repository:
```
git clone https://github.com/yanx27/JS3C-Net.git
```

Installation instructions for Ubuntu 16.04:
     
* Make sure <a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html">CUDA</a>  and <a href="https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html">cuDNN</a> are installed. Only this configurations has been tested: 
     - Python 3.6.9, Pytorch 1.3.1, CUDA 10.1;
* Compile the customized operators by `sh complile.sh` in `/lib`. 
* Install [spconv1.0](https://github.com/traveller59/spconv)  in `/lib/spconv`. We use the same version with [PointGroup](https://github.com/Jia-Research-Lab/PointGroup), you can install it according to the instruction. Higher version spconv may cause issues.


### Data Preparation
* SemanticKITTI and SemanticPOSS datasets can be found in [semantickitti-page](http://semantic-kitti.org/dataset.html#download) and [semanticposs-page](http://www.poss.pku.edu.cn/semanticposs.html). 
* Download the files related to **semantic segmentation** and extract everything into the same folder. 
* Use [voxelizer](https://github.com/jbehley/voxelizer) generate ground truths of **semantic scene completion**, where following parameters are used. We provide pre-processed SemanticPOSS SSC labels [here](https://drive.google.com/file/d/1AGagbRwQe3aR8liaC4YnkMW1iwSCLvvN/view?usp=sharing).
```angular2
min range: 2.5
max range: 70
future scans: 70
min extent: [0, -25.6, -2]
max extent: [51.2, 25.6,  4.4]
voxel size: 0.2
```

* Finally, the dataset folder should be organized as follows.
```angular2
SemanticKITTI(POSS)
├── dataset
│   ├── sequences
│   │  ├── 00
│   │  │  ├── labels
│   │  │  ├── velodyne
│   │  │  ├── voxels
│   │  │  ├── [OTHER FILES OR FOLDERS]
│   │  ├── 01
│   │  ├── ... ...

```
* Note that the data for official SemanticKITTI SSC benchmark only contains 1/5 of the whole sequence and they provide all extracted SSC data for the training set [here](http://semantic-kitti.org/assets/data_odometry_voxels_all.zip).
* (**New**) In this repo, we use old version of SemanticKITTI, and there is a bug of generating SSC data contains a wrong shift on upwards direction (see [issue](https://github.com/PRBonn/semantic-kitti-api/issues/49)). Therefore, we add an additional shifting to align their old version dataset [here](https://github.com/yanx27/JS3C-Net/blob/3433634c9cda7e8ed5c623e0ae9a9f2f2c5cee09/test_kitti_ssc.py#L94), and if you use the newest version of data, you can delete it. Also, you can check the alignment ratio by using `--debug`. 

### SemanticKITTI
#### Training
Run the following command to start the training. Output (logs) will be redirected to `./logs/JS3C-Net-kitti/`.  You can ignore this step if you want to use our pretrained model in `./logs/JS3C-Net-kitti/`.
```angular2
$ python train.py --gpu 0 --log_dir JS3C-Net-kitti --config opt/JS3C_default_kitti.yaml
```
#### Evaluation Semantic Segmentation
Run the following command to evaluate model on evaluation or test dataset
```
$ python test_kitti_segment.py --log_dir JS3C-Net-kitti --gpu 0 --dataset [val/test]
```

#### Evaluation Semantic Scene Completion
Run the following command to evaluate model on evaluation or test dataset
```
$ python test_kitti_ssc.py --log_dir JS3C-Net-kitti --gpu 0 --dataset [val/test]
```

### SemanticPOSS
Results on SemanticPOSS can be easily obtained by
```angular2
$ python train.py --gpu 0 --log_dir JS3C-Net-POSS --config opt/JS3C_default_POSS.yaml
$ python test_poss_segment.py --gpu 0 --log_dir JS3C-Net-POSS
```

## Pretrained Model
We trained our model on a single Nvidia Tesla V100 GPU with batch size 6. If you want to train on the TITAN GPU, you can choose batch size as 2. Please modify `dataset_dir` in `args.txt` to your path.

| Model | #Param | Segmentation | Completion | Checkpoint |
|--|--|--|--|--|
|JS3C-Net| 2.69M | 66.0 | 56.6 | [18.5MB](log/JS3C-Net-kitti) |

## Results on SemanticKITTI Benchmark
Quantitative results on **SemanticKITTI Benchmark** at the submisison time.
![](figure/benchmark.png)

## Acknowledgement
This project is not possible without multiple great opensourced codebases. 
* [SparseConv](https://github.com/facebookresearch/SparseConvNet)
* [spconv](https://github.com/traveller59/spconv)
* [PointGroup](https://github.com/Jia-Research-Lab/PointGroup)
* [nanoflann](https://github.com/jlblancoc/nanoflann)
* [semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api)
## License
This repository is released under MIT License (see LICENSE file for details).
