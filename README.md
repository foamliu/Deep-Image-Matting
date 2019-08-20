Just in case you are interested, [Deep Image Matting v2](https://github.com/foamliu/Deep-Image-Matting-v2) is an upgraded version of this.

# Deep Image Matting
This repository is to reproduce Deep Image Matting.

## Dependencies
- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow 1.9.0](https://www.tensorflow.org/)
- [Keras 2.1.6](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## Dataset
### Adobe Deep Image Matting Dataset
Follow the [instruction](https://sites.google.com/view/deepimagematting) to contact author for the dataset.

### MSCOCO
Go to [MSCOCO](http://cocodataset.org/#download) to download:
* [2014 Train images](http://images.cocodataset.org/zips/train2014.zip)


### PASCAL VOC
Go to [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) to download:
* VOC challenge 2008 [training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar)
* The test data for the [VOC2008 challenge](http://host.robots.ox.ac.uk/pascal/VOC/voc2008/index.html#testdata)

## ImageNet Pretrained Models
Download [VGG16](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5) into "models" folder.


## Usage
### Data Pre-processing
Extract training images:
```bash
$ python pre_process.py
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
Download pre-trained Deep Image Matting [Model](https://github.com/foamliu/Deep-Image-Matting/releases/download/v1.0/final.42-0.0398.hdf5) to "models" folder then run:
```bash
$ python demo.py
```

Image/Trimap | Output/GT | New BG/Compose | 
|---|---|---|
|![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/0_image.png)  | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/0_out.png)   | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/0_new_bg.png) |
|![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/0_trimap.png) | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/0_alpha.png) | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/0_compose.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/1_image.png)  | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/1_out.png)   | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/1_new_bg.png) | 
|![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/1_trimap.png) | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/1_alpha.png) | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/1_compose.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/2_image.png)  | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/2_out.png)   | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/2_new_bg.png) |
|![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/2_trimap.png) | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/2_alpha.png) | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/2_compose.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/3_image.png)  | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/3_out.png)   | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/3_new_bg.png) |
|![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/3_trimap.png) | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/3_alpha.png) | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/3_compose.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/4_image.png)  | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/4_out.png)   | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/4_new_bg.png) |
|![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/4_trimap.png) | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/4_alpha.png) | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/4_compose.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/5_image.png)  | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/5_out.png)   | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/5_new_bg.png) |
|![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/5_trimap.png) | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/5_alpha.png) | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/5_compose.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/6_image.png)  | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/6_out.png)   | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/6_new_bg.png) |
|![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/6_trimap.png) | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/6_alpha.png) | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/6_compose.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/7_image.png)  | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/7_out.png)   | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/7_new_bg.png) |
|![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/7_trimap.png) | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/7_alpha.png) | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/7_compose.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/8_image.png)  | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/8_out.png)   | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/8_new_bg.png) |
|![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/8_trimap.png) | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/8_alpha.png) | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/8_compose.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/9_image.png)  | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/9_out.png)   | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/9_new_bg.png) |
|![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/9_trimap.png) | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/9_alpha.png) | ![image](https://github.com/foamliu/Deep-Image-Matting/raw/master/images/9_compose.png)|

