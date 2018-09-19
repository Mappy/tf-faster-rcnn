# tf-faster-rcnn
A Tensorflow implementation of faster RCNN detection framework by Xinlei Chen (xinleic@cs.cmu.edu). This repository is based on the python Caffe implementation of faster RCNN available [here](https://github.com/endernewton/tf-faster-rcnn).

## Prerequisites
  - Docker installed on the machine
  - GPU with a RAM > 6Gb

## Nividia Installation

##### Nvidia drivers
- Install requirements: 
```sudo apt-get install build-essential linux-headers-$(uname -r)```
- Download CUDA from [NVIDIA website](https://developer.nvidia.com/cuda-downloads).
- Install

```
sudo dpkg -i FILE.deb (ex. : cuda-repo-ubuntu1604_8.0.61-1_amd64.deb)
sudo apt-get update
sudo apt-get install cuda nvidia-cuda-toolkit
```

- Download CuDNN from [NVIDIA website](https://developer.nvidia.com/cudnn) (ex.: Download cuDNN v6.0 (April 27, 2017), for CUDA 8.0)
- Extract and install files
```bash
cp include/* /usr/local/cuda-8.0/include/
cp lib64/* /usr/local/cuda-8.0/lib64/
```

- Restart computer
- Verify installation 

``` nvidia-smi ```

Fore more info check the following links :
- Nvidia [pdf guide](http://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Installation_Guide_Linux.pdf)
- Nvidia [CUDA Doc](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation)
- Nvidia [CUDA Doc](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation)

##### Docker-ce:
 - [Docker-ce](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/)
 - Ajout de l'utilisateur blur dans le groupe docker (sudo usermod -aG docker blur)

##### Nvidia Docker:
```bash
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
```

Test nvidia-smi : ```nvidia-docker run --rm nvidia/cuda nvidia-smi```

##### Using Dockerfile

Clone this repo and execute in the root of the project (i.e. tf-faster-rcnn directory) : 

``` nvidia-docker build --force-rm -t tf_faster_rcnn .```

## Project Installation

1. Create the working directory
```
mkdir <WORKING_DIR>
cd $WORKING_DIR
```
2. Clone Mappy Blur_ia project repository

```
git clone git@github.com:Mappy/tf-faster-rcnn.git
```

3. Update your -arch in setup script to match your GPU
  ```Shell
  cd tf-faster-rcnn/lib
  # Change the GPU architecture (-arch) if necessary
  vim setup.py
  ```

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |

  **Note**: You are welcome to contribute the settings on your end if you have made the code work properly on other GPUs. Also even if you are only using CPU tensorflow, GPU based code (for NMS) will be used by default, so please set **USE_GPU_NMS False** to get the correct output.


4. Build the docker image
  ```
  cd tf-faster-rcnn
  nvidia-docker build --force-rm -t tf_faster_rcnn .
  cd ..
  ```

### Setup database
Create the **database** directory to stock the traning and testing data 

```
cd $WORKING_DIR
mkdir database
cd database
```

#####  Installation of the Pascal VOC data for training and testing models

1. Download the training, validation, test data and VOCdevkit

```
mkdir VOCdevkit2007
cd VOCdevkit2007
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```
2. Extract all of these tars into the **VOCdevkit2007** directory
```
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
rm VOCtrainval_06-Nov-2007.tar VOCtest_06-Nov-2007.tar VOCdevkit_08-Jun-2007.tar
```
3. It should have this basic structure
```
VOCdevkit2007/                           # development kit
VOCdevkit2007/VOCcode/                   # VOC utility code
VOCdevkit2007/VOC2007                    # image sets, annotations, etc.
# ... and several other directories ...
```

#####  Installation of Mappy data

```
cd $WORKING_DIR/database
mkdir mappy
cd mappy
# copy mappy annoted data and the panoramic images here
```

### Setup pre-trained model
Create the **pre_trained_models** directory to stock the trained models 

```
cd $WORKING_DIR
mkdir pre_trained_models
cd pre_trained_models
```

##### ImageNet pre-trained models and weights
The current code support VGG16 and Resnet V1 models. Pre-trained models are provided by slim, you can get the pre-trained models here and set them in the data/imagenet_weights folder. For example for VGG16 model, you can set up like:

```
mkdir imagenet_weights
cd imagenet_weights
wget -v http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xzvf vgg_16_2016_08_28.tar.gz
mv vgg_16.ckpt vgg16.ckpt
cd ../..
```

For Resnet101, you can set up like:

```
cd pre_trained_models/imagenet_weights
wget -v http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
tar -xzvf resnet_v1_101_2016_08_28.tar.gz
mv resnet_v1_101.ckpt res101.ckpt
cd ../..
```

##### Pascal VOC pre-trained models

```
cd $WORKING_DIR/pre_trained_models
mkdir voc
cd voc
```

1. Download pre-trained model
  ```Shell
  # Resnet101 for voc pre-trained on 07+12 set
  ./data/scripts/fetch_faster_rcnn_models.sh
  ```
  **Note**: if you cannot download the models through the link, or you want to try more models, you can check out the following solutions and optionally update the downloading script:
  - Google drive [here](https://drive.google.com/open?id=0B1_fAEgxdnvJSmF3YUlZcHFqWTQ).

2. Create a folder to use the VOC pre-trained model
  ```Shell
  NET=res101
  TRAIN_IMDB=voc_2007_trainval+voc_2012_trainval
  mkdir -p $WORKING_DIR/pre_trained_models/voc/${NET}/${TRAIN_IMDB}
  cd $WORKING_DIR/pre_trained_models/voc/${NET}/${TRAIN_IMDB}
  mv ../../../data/voc_2007_trainval+voc_2012_trainval ./default
  ```

### Setup output directory
The output directory will be mounted with the output directory of the docker to persist the trained models
```
cd $WORKING_DIR
mkdir output
```

## Working with the project

### Test VOC pre trained model
1. Launch the docker with the mounted directories
```
cd $WORKING_DIR/tf-faster-rcnn
nvidia-docker run --rm -it \
 -v /home/blur/dev/ia/tf-faster-rcnn/data:/ai/tf-faster-rcnn/data \
 -v /home/blur/dev/ia/database/VOCdevkit2007:/ai/tf-faster-rcnn/data/VOCdevkit2007 \
 -v /home/blur/dev/ia/pre_trained_models/voc:/ai/tf-faster-rcnn/output \
 --name tf_faster_rcnn_name tf_faster_rcnn
```

2. Demo for testing on custom images
```
GPU_ID=0
CUDA_VISIBLE_DEVICES=${GPU_ID} ./tools/demo.py
```
**Note**: Resnet101 testing probably requires several gigabytes of memory, so if you encounter memory capacity issues, please install it with CPU support only. Refer to [Issue 25](https://github.com/endernewton/tf-faster-rcnn/issues/25).

3. Test with pre-trained Resnet101 models
  ```Shell
  GPU_ID=0
  ./experiments/scripts/test_faster_rcnn.sh $GPU_ID pascal_voc_0712 res101
  ```
  **Note**: If you cannot get the reported numbers (79.8 on my side), then probably the NMS function is compiled improperly, refer to [Issue 5](https://github.com/endernewton/tf-faster-rcnn/issues/5).

### Train your own model

##### Training and test the VOC with pre-trained imagenets model
1. Launch the docker with the mounted directories
```
cd $WORKING_DIR/tf-faster-rcnn
nvidia-docker run --rm -it \
 -v /home/blur/dev/ia/tf-faster-rcnn/data:/ai/tf-faster-rcnn/data \
 -v /home/blur/dev/ia/pre_trained_models/imagenet_weights:/ai/tf-faster-rcnn/data/imagenet_weights \
 -v /home/blur/dev/ia/database/VOCdevkit2007:/ai/tf-faster-rcnn/data/VOCdevkit2007 \
 -v /home/blur/dev/ia/output:/ai/tf-faster-rcnn/output \
 --name tf_faster_rcnn_name tf_faster_rcnn
```

2. Train
  ```Shell
  ./experiments/scripts/train_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
  # GPU_ID is the GPU you want to test on
  # NET in {vgg16, res50, res101, res152} is the network arch to use
  # DATASET {pascal_voc, pascal_voc_0712, coco} is defined in train_faster_rcnn.sh
  # Examples:
  ./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc res101
  ```
  **Note**: Please double check you have deleted soft link to the pre-trained models before training. If you find NaNs during training, please refer to [Issue 86](https://github.com/endernewton/tf-faster-rcnn/issues/86). Also if you want to have multi-gpu support, check out [Issue 121](https://github.com/endernewton/tf-faster-rcnn/issues/121).

4. Test and evaluate
  ```Shell
  ./experiments/scripts/test_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
  # GPU_ID is the GPU you want to test on
  # NET in {vgg16, res50, res101, res152} is the network arch to use
  # DATASET {pascal_voc, pascal_voc_0712, coco} is defined in test_faster_rcnn.sh
  # Examples:
  ./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc res101
  ```

##### Training and test the Mappy with pre-trained imagenets model
1. Launch the docker with the mounted directories
```
cd $WORKING_DIR/tf-faster-rcnn
nvidia-docker run --rm -it \
 -v /home/blur/dev/ia/tf-faster-rcnn/data:/ai/tf-faster-rcnn/data \
 -v /home/blur/dev/ia/pre_trained_models/imagenet_weights:/ai/tf-faster-rcnn/data/imagenet_weights \
 -v /home/blur/dev/ia/database/mappy:/ai/tf-faster-rcnn/data/mappy \
 -v /home/blur/dev/ia/output:/ai/tf-faster-rcnn/output \
 --name tf_faster_rcnn_name tf_faster_rcnn
```

2. Train
  ```Shell
  ./experiments/scripts/train_faster_rcnn.sh [GPU_ID] [DATASET] [NET] [ITERATIONS]
  # GPU_ID is the GPU you want to test on
  # NET in {vgg16, res50, res101, res152} is the network arch to use
  # DATASET {pascal_voc, pascal_voc_0712, coco} is defined in train_faster_rcnn.sh
  # Examples:
  ./experiments/scripts/train_faster_rcnn.sh 0 mappy res101 110000
  ```
  **Note**: Please double check you have deleted soft link to the pre-trained models before training. If you find NaNs during training, please refer to [Issue 86](https://github.com/endernewton/tf-faster-rcnn/issues/86). Also if you want to have multi-gpu support, check out [Issue 121](https://github.com/endernewton/tf-faster-rcnn/issues/121).

4. Test and evaluate
  ```Shell
  ./experiments/scripts/test_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
  # GPU_ID is the GPU you want to test on
  # NET in {vgg16, res50, res101, res152} is the network arch to use
  # DATASET {pascal_voc, pascal_voc_0712, coco} is defined in test_faster_rcnn.sh
  # Examples:
  ./experiments/scripts/test_faster_rcnn.sh 0 mappy res101 110000
  ```


By default, trained networks are saved under:

```
output/[NET]/[DATASET]/default/
```

Test outputs are saved under:

```
output/[NET]/[DATASET]/default/[SNAPSHOT]/
```

Tensorboard information for train and validation is saved under:

```
tensorboard/[NET]/[DATASET]/default/
tensorboard/[NET]/[DATASET]/default_val/
```

The default number of training iterations is kept the same to the original faster RCNN for VOC 2007, however I find it is beneficial to train longer (see [report](https://arxiv.org/pdf/1702.02138.pdf) for COCO), probably due to the fact that the image batch size is one. For VOC 07+12 we switch to a 80k/110k schedule following [R-FCN](https://github.com/daijifeng001/R-FCN). Also note that due to the nondeterministic nature of the current implementation, the performance can vary a bit, but in general it should be within ~1% of the reported numbers for VOC, and ~0.2% of the reported numbers for COCO. Suggestions/Contributions are welcome.

### Citation
If you find this implementation or the analysis conducted in our report helpful, please consider citing:

    @article{chen17implementation,
        Author = {Xinlei Chen and Abhinav Gupta},
        Title = {An Implementation of Faster RCNN with Study for Region Sampling},
        Journal = {arXiv preprint arXiv:1702.02138},
        Year = {2017}
    }
    
Or for a formal paper, [Spatial Memory Network](https://arxiv.org/abs/1704.04224):

    @article{chen2017spatial,
      title={Spatial Memory for Context Reasoning in Object Detection},
      author={Chen, Xinlei and Gupta, Abhinav},
      journal={arXiv preprint arXiv:1704.04224},
      year={2017}
    }

For convenience, here is the faster RCNN citation:

    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }
