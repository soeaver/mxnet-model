## CLS (Classification)

Please install official MXNet(https://github.com/apache/incubator-mxnet) for evaluating and finetuning.

### Disclaimer

Most of the models are converted from other projects, the main contribution belongs to the original authors.

Project links:

[pytorch-classification](https://github.com/soeaver/pytorch-classification)


### Performance on imagenet validation.
**1. Top-1/5 error of pre-train models in this repository.**

 Network|224/299<br/>(single-crop)|224/299<br/>(12-crop)|320/395<br/>(single-crop)|320/395<br/>(12-crop)
 :---:|:---:|:---:|:---:|:---:
 resnet18-priv| 29.11/10.07 | 26.69/8.64 | 27.54/8.98 | 26.23/8.21

 - The resnet18-priv is trained under [pytorch](https://github.com/soeaver/pytorch-classification) by bupt-priv.

### Check the performance
**1. Download the ILSVRC 2012 classification val set [6.3GB](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar), and put the extracted images into the directory:**

      ~/Database/ILSVRC2012

**2. Modify the parameter settings**

 Network|val_file|mean_value|std
 :---:|:---:|:---:|:---:
 resnet18-priv | ILSVRC2012_val | [103.52, 116.28, 123.675] | [57.375, 57.12, 58.395]

**3. then run evaluation_cls.py**

    python evaluation_cls.py
