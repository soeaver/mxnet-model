## CLS (Classification)

Please install official MXNet(https://github.com/apache/incubator-mxnet) for evaluating and finetuning.

### Disclaimer

Most of the models are converted from other projects, the main contribution belongs to the original authors.

Project links:

[mxnet-model-gallery](https://github.com/dmlc/mxnet-model-gallery)、 [tensorflow slim](https://github.com/tensorflow/models/tree/master/slim)、 [craftGBD](https://github.com/craftGBD/craftGBD)、 [ResNeXt](https://github.com/facebookresearch/ResNeXt)、 [DenseNet](https://github.com/liuzhuang13/DenseNet)、 [wide-residual-networks](https://github.com/szagoruyko/wide-residual-networks)、 [keras deep-learning-models](https://github.com/fchollet/deep-learning-models)、 [ademxapp](https://github.com/itijyou/ademxapp)、 [DPNs](https://github.com/cypw/DPNs)


### Performance on imagenet validation.
**1. Top-1/5 error of pre-train models in this repository.**

 Network|224/299<br/>(single-crop)|224/299<br/>(12-crop)|320/395<br/>(single-crop)|320/395<br/>(12-crop)
 :---:|:---:|:---:|:---:|:---:
 resnet18-priv| 29.11/10.07 | 26.69/8.64 | 27.54/8.98 | 26.23/8.21

 - The resnet18-priv, resnext26-32x4d-priv are trained under [pytorch](https://github.com/soeaver/pytorch-classification) by bupt-priv.

### Check the performance
**1. Download the ILSVRC 2012 classification val set [6.3GB](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar), and put the extracted images into the directory:**

      ~/Database/ILSVRC2012

**2. Modify the parameter settings**

 Network|val_file|mean_value|std
 :---:|:---:|:---:|:---:
 resnet-v2(101/152/269)| ILSVRC2012_val | [102.98, 115.947, 122.772] | [1.0, 1.0, 1.0]
 resnet18-priv, resnext26-32x4d-priv<br/>resnet38a, resnext50-32x4d<br/>resnext101-32x4d, resnext101-64x4d<br/>wrn50-2, air(x) | ILSVRC2012_val | [103.52, 116.28, 123.675] | [57.375, 57.12, 58.395]
 inception-v3| **ILSVRC2015_val** | [128.0, 128.0, 128.0] | [128.0, 128.0, 128.0] 
 inception-v2, xception<br/>inception-v4, inception-resnet-v2 | ILSVRC2012_val | [128.0, 128.0, 128.0] | [128.0, 128.0, 128.0] 
 dpn(92/98/131/107)| ILSVRC2012_val | [104.0, 117.0, 124.0] | [59.88, 59.88, 59.88]


**3. then run evaluation_cls.py**

    python evaluation_cls.py
