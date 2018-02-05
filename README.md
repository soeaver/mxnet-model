# mxnet-model
MXNet models (only classification for now) and deploy prototxt for resnet, resnext, inception_v3, inception_v4, inception_resnet, wider_resnet, densenet, DPNs and other networks.

## We recommend using these MXNet models with official MXNet repository
Please install [MXNet](https://github.com/apache/incubator-mxnet) for evaluating and finetuning.

## Disclaimer

Most of the pre-train models are converted from other projects, the main contribution belongs to the original authors.

Project links:

[pytorch-classification](https://github.com/soeaver/pytorch-classification)、[mxnet-model-gallery](https:
//github.com/dmlc/mxnet-model-gallery)、 [tensorflow slim](https://github.com/tensorflow/models/tree/master/slim)
、 [craftGBD](https://github.com/craftGBD/craftGBD)、 [ResNeXt](https://github.com/facebookresearch/ResNeXt)、
[keras deep-learning-models](https://github.com/fchollet/deep-learning-models)、 [DPNS](https://github.com/cypw/DPNs)、
[Senet](https://github.com/hujie-frank/SENet)


## CLS (Classification, more details are in [cls](https://github.com/LeonJWH/mxnet-model/tree/master/cls))
### Performance on imagenet validation.
**Top-1/5 error of pre-train models in this repository (Pre-train models download [urls](https://github.com/LeonJWH/mxnet-model/tree/master/cls#performance-on-imagenet-validation)).**

Network|224/299<br/>(single-crop)|320/395<br/>(single-crop)| Download
 :---:|:---:|:---:|:---:
 resnet101_v2        | 21.89/6.01  | 20.44/5.22 | [170.3MB](https://pan.baidu.com/s/1i6wU0lv)(BaiduCloud)
 resnet152_v2        | 20.71/5.42  | 19.65/4.75 | [230.2MB](https://pan.baidu.com/s/1c3KLlIS)(BaiduCloud)
 resnet269_v2        | 19.73/4.99  | 18.64/4.33 | [390.4MB](https://pan.baidu.com/s/1brllCPT)(BaiduCloud)
 resnext50_32x4d     | 22.38/6.31  | 21.10/5.52 | [95.8MB](https://pan.baidu.com/s/1pMO26in)(BaiduCloud)
 resnext101_32x4d    | 21.33/5.80  | 19.92/4.97 | [169.1MB](https://pan.baidu.com/s/1hsTkrYW)(BaiduCloud)
 resnext101_64x4d    | 20.60/5.41  | 19.26/4.63 | [391.2MB](https://pan.baidu.com/s/1eTn7hqq)(BaiduCloud)
 inception_v3        | 21.70/5.75  |     ..     | [91.1MB](https://pan.baidu.com/s/1jKcjTSM)(BaiduCloud)
 inception_v4        | 20.03/5.09  |     ..     | [163.1MB](https://pan.baidu.com/s/1snp2NG5)(BaiduCloud)
 inception_resnet_v2 | 19.86/4.83  |     ..     | [213.3MB](https://pan.baidu.com/s/1ei5PxG)(BaiduCloud)
 xception            | 20.89/5.48  |     ..     | [87.4MB](https://pan.baidu.com/s/1dG5QdLR)(BaiduCloud)
 air101              | 21.32/5.77  |     ..     | [246.2MB](https://pan.baidu.com/s/1o94RK6Y)(BaiduCloud)
 dpn-92-extra        | 19.98/5.06  | 19.00/4.37 | [145MB](https://goo.gl/1sbov7)(GoogleDrive)
 dpn107-extra        | 19.75/4.94  | 18.34/4.19 | [333MB](https://goo.gl/YtokAb)(GoogleDrive)
 se-resnet50         | 22.39/6.37  | 20.50/5.23 | [107.4MB](https://pan.baidu.com/s/1c39VQJQ)(BaiduCloud)
 se-resnext50-32x4d  | 20.96/5/54  | 19.35/4.66 | [105.4MB](https://pan.baidu.com/s/1qZ6P9DE)(BaiduCloud)
 se-inception-v2     | 23.64/7.05  | 21.61/5.88 | [45.6MB](https://pan.baidu.com/s/1htqkSh2)(BaiduCloud)

 - The resnet18-priv, resnext26-32x4d-priv is trained under [pytorch](https://github.com/soeaver/pytorch-classification) by bupt-priv.
 - 224x224(base_size=256) and 320x320(base_size=320) crop size for resnet-v2/resnext, 299x299(base_size=320) and 395x395(base_size=395) crop size for inception.
