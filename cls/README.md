## CLS (Classification)

Please install official MXNet(https://github.com/apache/incubator-mxnet) for evaluating and finetuning.

### Disclaimer

Most of the models are converted from other projects, the main contribution belongs to the original authors.

Project links:

[pytorch-classification](https://github.com/soeaver/pytorch-classification)、[mxnet-model-gallery](https://github.com/dmlc/mxnet-model-gallery)、 [tensorflow slim](https://github.com/tensorflow/models/tree/master/slim)、 [craftGBD](https://github.com/craftGBD/craftGBD)、 [ResNeXt](https://github.com/facebookresearch/ResNeXt)、 [keras deep-learning-models](https://github.com/fchollet/deep-learning-models)、 [DPNS](https://github.com/cypw/DPNs)、 [Senet](https://github.com/hujie-frank/SENet)

### Performance on imagenet validation.
**1. Top-1/5 error of pre-train models in this repository.**

 Network|224/299<br/>(single-crop)|224/299<br/>(12-crop)|320/395<br/>(single-crop)|320/395<br/>(12-crop)
 :---:|:---:|:---:|:---:|:---:
 resnet18-priv| 29.11/10.07 | 26.69/8.64 | 27.54/8.98 | 26.23/8.21
 resnet101_v2| 21.89/6.01  |            | 20.44/5.22 | 
 resnet152_v2| 20.71/5.42  |            | 19.65/4.75 | 
 resnet269_v2| 19.73/4.99  |            | 18.64/4.33 | 
 resnext26_32x4d-priv| 25.62/8.12  |            | 24.21/7.22 | 
 resnext50_32x4d| 22.38/6.31  |            | 21.10/5.52 | 
 resnext101_32x4d| 21.33/5.80  |            | 19.92/4.97 | 
 resnext101_64x4d| 20.60/5.41  |            | 19.26/4.63 | 
 inception_v1_tf| 29.56/10.01 |            |            |
 inception_v3| 21.70/5.75  |            |            |
 inception_v4| 20.03/5.09  |            |            |
 air101| 21.32/5.77 |           |           |
 xception| 20.89/5.48 |           |           |
 dpn-68-extra| 22.45/6.09 |        | 20.92/5.26 |        |
 dpn-92-extra| 19.98/5.06 |        | 19.00/4.37 |        |
 dpn-98| 20.15/5.15 |      | 18.94/4.44 |       |
 dpn131| 19.93/5.12	|      | 18.62	4.23 |       |
 dpn107-extra| 19.75/4.94	|        |18.34/4.19 |       |
 se-resnet50| 22.39/6.37 |         |           |       |


 - The resnet18-priv, resnext26-32x4d-priv is trained under [pytorch](https://github.com/soeaver/pytorch-classification) by bupt-priv.
 - 224x224(base_size=256) and 320x320(base_size=320) crop size for resnet-v2/resnext, 299x299(base_size=320) and 395x395(base_size=395) crop size for inception.

### Check the performance
**1. Download the ILSVRC 2012 classification val set [6.3GB](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar), and put the extracted images into the directory:**

      ~/Database/ILSVRC2012

**2. Modify the parameter settings**

 Network|val_file|mean_value|std
 :---:|:---:|:---:|:---:
 resnet-v2(101/152/269)| ILSVRC2012_val | [102.98, 115.947, 122.772] | [1.0, 1.0, 1.0]
 resnet18-priv, resnext26-32x4d-priv<br/>resnext50-32x4d, resnext101-32x4d<br/>resnext101-64x4d, air(x) | ILSVRC2012_val | [103.52, 116.28, 123.675] | [57.375, 57.12, 58.395]
 inception-v3 | ILSVRC2015_val | [128.0, 128.0, 128.0] | [128.0, 128.0, 128.0]
 inception-v2, xception<br/>inception-v4 | ILSVRC2012_val | [128.0, 128.0, 128.0] | [128.0, 128.0, 128.0]
 dpn(68/92/98/131/107)	| ILSVRC2012_val	| [104.0, 117.0, 124.0]	| [59.88, 59.88, 59.88]
 
**3. then run evaluation_cls.py**

    python evaluation_cls.py
