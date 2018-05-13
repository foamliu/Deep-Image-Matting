# Deep Image Matting
This repository is to reproduce Deep Image Matting.

## Dependencies
- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## Dataset
### Adobe Deep Image Matting Dataset
Follow the [instruction](https://sites.google.com/view/deepimagematting) to contact author for the dataset.

### MSCOCO
Go to [MSCOCO](http://cocodataset.org/#download) to download:
* [2014 Train images](http://images.cocodataset.org/zips/train2014.zip)


### PASCAL VOC
Go to [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) to download:
* VOC challenge 2007 [training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)

## ImageNet Pretrained Models
Download [VGG16](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5) into models folder.


## Usage
### Data Pre-processing
Extract training images:
```bash
$ python pre-process.py
```

### Train
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### Demo

```bash
$ python demo.py
```

Image | Trimap | Output | GT |
|---|---|---|---|
|![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/182_18293_image.png)  | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/182_18293_trimap.png) | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/182_18293_out.png)| ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/182_18293_alpha.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/120_12081_image.png)  | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/120_12081_trimap.png) | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/120_12081_out.png)| ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/120_12081_alpha.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/249_24980_image.png)  | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/249_24980_trimap.png) | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/249_24980_out.png)| ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/249_24980_alpha.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/image.png)  | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/trimap.png) | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/out.png)| |
