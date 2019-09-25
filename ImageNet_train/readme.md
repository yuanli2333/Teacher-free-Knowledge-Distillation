# Tf-KD on ImageNet.

Updating.....

Our implementation is based on the official example of PyTorch: [ImageNet training in PyTorch](https://github.com/pytorch/examples/tree/master/imagenet).

## 1. Requirements
Dataset pre-processing, refer to [here](https://github.com/pytorch/examples/tree/master/imagenet#requirements)

Environment, refer to [here](https://github.com/yuanli2333/Teacher-free-Knowledge-Distillation#11-environment)

## 2. Tf-KD: self-training
ResNet18 self-training:
```
python main.py -a resnet18 -at resnet18 --KD 1 
```

ResNet18 taught by DenseNet121:
```
python main.py -a resnet18 --at densenet121 --KD 1
```

ResNeXt101 self-training:
```
python main.py -a resnext101_32x8d -at resnext101_32x8d --KD 1
```

## 3. Tf-KD: manually-designed teacher(regularization)

Train ResNet18
```
python main.py -a resenet18 --regularization 1
```

Train ResNeXt101:
```
python main.py -a resnext101_32x8d --regularization 1
```

## 4. Lable Smoothing Regularization (LSR)

ResNet18 LSR:
```
python main.py -a resnet18 --smoothing 1
```

DenseNet121 LSR:
```
python main.py -a densenet121 --smoothing 1
```
