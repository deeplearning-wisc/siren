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

Download the processed VOC 2007 and 2012 dataset from [here](https://drive.google.com/file/d/1n9C4CiBURMSCZy2LStBQTzR17rD_a67e/view?usp=sharing).

The VOC dataset folder should have the following structure:
<br>

     └── VOC_DATASET_ROOT
         |
         ├── JPEGImages
         ├── voc0712_train_all.json
         └── val_coco_format.json

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

Before training, 1) modify the file address for saving the checkpoint by changing "EXP_DIR" in the shell files inside ./configs/; 2) modify the address for the training and ood dataset in the main.py file.