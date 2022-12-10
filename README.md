# SIREN

This is the source code accompanying the paper [***SIREN: Shaping Representations for Detecting Out-of-Distribution Objects***](https://openreview.net/forum?id=8E8tgnYlmN) by Xuefeng Du, Gabriel Gozum, Yifei Ming, and Yixuan Li


The codebase is heavily based on [DETReg](https://github.com/amirbar/DETReg).

## Ads 

Checkout our ICLR'22 work [VOS](https://github.com/deeplearning-wisc/vos) and CVPR'22 work [STUD](https://github.com/deeplearning-wisc/stud) on OOD detection for faster R-CNN models if you are interested!

## Requirements
```
pip install -r requirements.txt
```
In addition, install detectron2 following [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

## Dataset Preparation

**PASCAL VOC**



Download [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset (2012trainval, 2007trainval, and 2007test):
```bash
mkdir VOC_DATASET_ROOT
cd VOC_DATASET_ROOT
wget http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar -xvf VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
```

The VOC dataset folder should have the following structure:
<br>

     └── VOC_DATASET_ROOT
         |
         ├── VOCdevkit/
              ├── VOC2007
              └── VOC2012

**COCO**

Download COCO2017 dataset from the [official website](https://cocodataset.org/#home). 

Download the OOD dataset (json file) when the in-distribution dataset is Pascal VOC from [here](https://drive.google.com/file/d/1Wsg9yBcrTt2UlgBcf7lMKCw19fPXpESF/view?usp=sharing). 

Download the OOD dataset (json file) when the in-distribution dataset is BDD-100k from [here](https://drive.google.com/file/d/1AOYAJC5Z5NzrLl5IIJbZD4bbrZpo0XPh/view?usp=sharing).

Put the two processed OOD json files to ./anntoations

The COCO dataset folder should have the following structure:
<br>

     └── COCO_DATASET_ROOT
         |
         ├── annotations
            ├── xxx (the original json files)
            ├── instances_val2017_ood_wrt_bdd_rm_overlap.json
            └── instances_val2017_ood_rm_overlap.json
         ├── train2017
         └── val2017

**BDD-100k**

Donwload the BDD-100k images from the [official website](https://bdd-data.berkeley.edu/).

Download the processed BDD-100k json files from [here](https://drive.google.com/file/d/1ZbbdKEakSjyOci7Ggm046hCCGYqIHcbE/view?usp=sharing) and [here](https://drive.google.com/file/d/1Rxb9-6BUUGZ_VsNZy9S2pWM8Q5goxrXY/view?usp=sharing).

The BDD dataset folder should have the following structure:
<br>

     └── BDD_DATASET_ROOT
         |
         ├── images
         ├── val_bdd_converted.json
         └── train_bdd_converted.json
**OpenImages**

Download our OpenImages validation splits [here](https://drive.google.com/file/d/1UPuxoE1ZqCfCZX48H7bWX7GGIJsTUrt5/view?usp=sharing). We created a tarball that contains the out-of-distribution data splits used in our paper for hyperparameter tuning. Do not modify or rename the internal folders as those paths are hard coded in the dataset reader. The OpenImages dataset is created in a similar way following this [paper](https://openreview.net/forum?id=YLewtnvKgR7&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2021%2FConference%2FAuthors%23your-submissions)). 

The OpenImages dataset folder should have the following structure:
<br>

     └── OEPNIMAGES_DATASET_ROOT
         |
         ├── coco_classes
         └── ood_classes_rm_overlap



**Visualization of the OOD datasets**

 The OOD images with respect to different in-distribution datasets can be downloaded from [ID-VOC-OOD-COCO](https://drive.google.com/drive/folders/1NxodhoxTX5YBHJWHAa6tB2Ta1oxoTfzu?usp=sharing), [ID-VOC-OOD-openimages](https://drive.google.com/drive/folders/1pRP7CAWG7naDECfejo03cl7PF3VJEjrn?usp=sharing), [ID-BDD-OOD-COCO](https://drive.google.com/drive/folders/1Wgmfcp2Gd3YvYVyoRRBWUiwwKYXJeuo8?usp=sharing), [ID-BDD-OOD-openimages](https://drive.google.com/drive/folders/1LyOFqSm2G8x7d2xUkXma2pFJVOLgm3IQ?usp=sharing).


## Training

Firstly, enter the deformable detr folder by running
```
cd detr
```

### address change
Before training, 1) modify the file address for saving the checkpoint by changing "EXP_DIR" in the shell files inside ./configs/; 2) modify the address for the training and ood dataset in the main.py file.

### compile cuda functions

```
cd models/ops & python setup.py build install & cd ../../
```

**Vanilla Faster-RCNN with VOC as the in-distribution dataset**
```
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/voc/vanilla.sh voc_id
```
**Vanilla Faster-RCNN with BDD as the in-distribution dataset**
```
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/bdd/vanilla.sh voc_id
```
**SIREN on VOC**
```
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/voc/siren.sh voc_id
```

## Evaluation

**Evaluation with the in-distribution dataset to be VOC**

Firstly run on the in-distribution dataset:
```
./configs/voc/<config file>.sh voc_id --resume snapshots/voc/<config file>/checkpoint.pth --eval
```
Then run on the in-distribution training dataset (not required for vMF score):
```
./configs/voc/<config file>.sh voc_id --resume snapshots/voc/<config file>/checkpoint.pth --eval --maha_train
```

Finally run on the OOD dataset of COCO:
```
./configs/voc/<config file>.sh coco_ood --resume snapshots/voc/<config file>/checkpoint.pth --eval
```
Obtain the metrics by vMF and KNN score using:
```
python voc_coco_vmf.py --name xxx --pro_length xx --use_trained_params 1
```
"name" means the vanilla or siren

"pro_length" means the dimension of projected space. We use 16 for VOC and 64 for BDD.

"use_trained_params" denotes whether we use the learned vMF distributions for OOD detection.



**Pretrained models**

The pretrained models for Pascal-VOC can be downloaded from [vanilla](https://drive.google.com/file/d/1-9ssnAL4UPv4sOpm8-jfrqgPMIbZ82NV/view?usp=sharing) and [SIREN](https://drive.google.com/file/d/1ZUr-ytjtDOYHfeM1geE_MI5B0rY0aisa/view?usp=sharing).

The pretrained models for BDD-100k can be downloaded from [vanilla](https://drive.google.com/file/d/1O_EoEQMSNDMBrAVn0Opr56BY_lSS-1P-/view?usp=sharing) and [SIREN](https://drive.google.com/file/d/1QOEMAUj0E9KWNM9e-jmlNw2LjTQv2om8/view?usp=sharing).


## SIREN on Faster R-CNN models

Following [VOS](https://github.com/deeplearning-wisc/vos) for intallation and preparation.

**SIREN on VOC**
```
python train_net_gmm.py 
--dataset-dir path/to/dataset/dir
--num-gpus 8 
--config-file VOC-Detection/faster-rcnn/center64_0.1.yaml
--random-seed 0 
--resume
```

## Evaluation

**Evaluation with the in-distribution dataset to be VOC**

Firstly run on the in-distribution dataset:
```
python apply_net.py 
--dataset-dir path/to/dataset/dir
--test-dataset voc_custom_val 
--config-file VOC-Detection/faster-rcnn/center64_0.1.yaml 
--inference-config Inference/standard_nms.yaml 
--random-seed 0 
--image-corruption-level 0 
--visualize 0
```
Then run on the in-distribution training dataset (not required for vMF score):
```
python apply_net.py 
--dataset-dir path/to/dataset/dir
--test-dataset voc_custom_train 
--config-file VOC-Detection/faster-rcnn/center64_0.1.yaml 
--inference-config Inference/standard_nms.yaml 
--random-seed 0 
--image-corruption-level 0 
--visualize 0
```

Finally run on the OOD dataset:

```
python apply_net.py
--dataset-dir path/to/dataset/dir
--test-dataset coco_ood_val 
--config-file VOC-Detection/faster-rcnn/center64_0.1.yaml 
--inference-config Inference/standard_nms.yaml 
--random-seed 0 
--image-corruption-level 0 
--visualize 0
```
Obtain the metrics by both vMF and KNN score using:
```
python voc_coco_vmf.py --name center64_0.1 --thres xxx 
```
```
python voc_coco_knn.py --name center64_0.1 --thres xxx
```

Here the threshold is determined according to [ProbDet](https://github.com/asharakeh/probdet). It will be displayed in the screen as you finish evaluating on the in-distribution dataset.

**Pretrained models**

The pretrained models for Pascal-VOC can be downloaded from [SIREN-ResNet](https://drive.google.com/file/d/1o5yhkgPUhmlp9Co1DCuh_li6kwATUoET/view?usp=sharing) with the projected dimension to be 64.


## Citation ##
If you found any part of this code is useful in your research, please consider citing our paper:

```
@inproceedings{du2022siren,
  title={SIREN: Shaping Representations for Detecting Out-of-distribution Objects},
  author={Du, Xuefeng and Gozum, Gabriel and Ming, Yifei and Li, Yixuan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```
