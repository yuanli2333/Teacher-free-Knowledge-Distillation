# Teacher-free-Knowledge-Distillation
## Implementation for our paper: Revisiting Knowledge Distillation via Label Smoothing Regularization, [arxiv](https://arxiv.org/abs/1909.11723)

Our work suggests that: when a neural network is too powerful to find stronger teacher models, or computation resource is limited to train teacher models, "self-training" or "manually-designed regularization" can be applied. 

For example, ResNeXt101-32x8d is a powerful model with 88.79M parameters and 16.51G FLOPs on ImageNet, and it is hard or computation expensive to train a stronger teacher model for this student. Our strategy can further improve this powerful student model by 0.48\% without extra computation on ImageNet. Similarly, when taking a powerful single model ResNeXt29-8x64d with 34.53M parameters as a student model, our self-training implementation achieves more than 1.0\% improvement on CIFAR100 (from 81.03\% to 82.08\%).


![](/figures/figure_ill.png)



## 1. Preparations

Clone this repository:
```
git clone https://github.com/yuanli2333/Teacher-free-Knowledge-Distillation.git
```

### 1.1 Environment
Build a new environment and install:
```
pip install -r requirements.txt
```

Better use: NVIDIA GPU + CUDA9.0 + Pytorch 1.2.0

Please do not use other versions of pytorch, otherwise, some experiment results may not be reproduced because some slight difference would make the hyper-parameters different. 

### 1.2 Dataset
[CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html), [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) and [Tiny_ImageNet](https://tiny-imagenet.herokuapp.com/);
For CIFAR100 and CIFAR10, our codes will download the datasets automatically. For Tiny-ImageNet, you should download and put in the dir: "data/". The follow instruction and commands are for CIFAR100.

## 2. Train baseline models
You can skip this step by using our pre-trained models in [here](https://drive.google.com/open?id=1TMZ-TSbB_OanKpXupIqvYdYpw8p7tVti). Download and unzip to: experiments/pretrained_teacher_models/


Use ''--model_dir'' to specify the directory of "parameters", model saving and log saving.

For example, normally train ResNet18 to obtain the pre-trained teacher:
```
CUDA_VISIBLE_DEVICES=0 python main.py --model_dir experiments/base_experiments/base_resnet18/
```
We ignore the command ''CUDA_VISIBLE_DEVICES=gpu_id'' in the following commands

Normally train MobileNetV2 to obtain the baseline model and baseline accuracy:
```
python main.py --model_dir experiments/base_experiments/base_mobilenetv2/
```

Normally train ResNeXt29 to obtain the baseline model and baseline accuracy:
```
python main.py --model_dir experiments/base_experiments/base_resnext29/
```
The baseline accuracy (in %) on CIFAR100 is:

| Model  |  Baseline Acc |
| :---     |    :---:   | 
| MobileNetV2 |    68.38   |
| ShuffleNetV2  |    70.34  |   
| ResNet18 |    75.87   |   
| ResNet50 |    78.16   |  
| GoogLeNet |    78.72  | 
| Desenet121 |   79.04  |   
| ResNeXt29 |   81.03  |   
 
  

## 3. Exploratory experiments (Section 2 in our paper)


### 3.1 Reversed KD (Re-KD)
Normal KD: ResNet18 teach MobileNetV2
```
python main.py --model_dir experiments/kd_experiments/mobilenet_v2_distill/resnet18_teacher/
```


Re-KD: MobileNetV2 teach ResNet18
```
python main.py --model_dir experiments/kd_experiments/resnet18_distill/mobilenet_v2_teacher/
```

Re-KD: ShuffleNetV2 teach ResNet18
```
python main.py --model_dir experiments/kd_experiments/resnet18_distill/shufflenet_v2_teacher/
```

Re-KD experiment results on CIFAR100: 

![](/figures/Re-KD.png)

### 3.2 Defective KD (De-KD)
Use the arguments "--pt_teacher" to switch to Defective KD experiment.

For expample, use a pooly-trained Teacher ResNet18 with 15.48% accuracy (just trained one epoch) to teach MobileNetV2:

```
python main.py --model_dir experiments/kd_experiments/mobilenet_v2_distill/resnet18_teacher/ --pt_teacher
```

Use one-epoch trained teacher ResNet18 to teach ShuffleNetV2:
```
python main.py --model_dir experiments/kd_experiments/shufflenet_v2_distill/resnet18_teacher/ --pt_teacher
```

Use 50-epoch-trained teacher ResNet50 (acc:45.82%) to teach ShuffleNetV2:
```
python main.py --model_dir experiments/kd_experiments/shufflenet_v2_distill/resnet50_teacher/ --pt_teacher
```
Use 50-epoch-trained teacher ResNet50 (acc:45.82%) to teach ResNet18:
```
python main.py --model_dir experiments/kd_experiments/resnet18_distill/resnet50_teacher/ --pt_teacher
```
Use 50-epoch-trained teacher ResNeXt29 (acc:51.94%) to teach MobileNetV2:
```
python main.py --model_dir experiments/kd_experiments/mobilenet_v2_distill/resnext29_teacher/ --pt_teacher
```
De-KD experiment results on CIFAR100: 
![](/figures/De-KD.png)

## 4. Teacher-free KD (Tf-KD)  (Section 5 in our paper)
We have two implementations to achieve Tf-KD, the first one is self-training, the second one is manually-designed teacher(regularization)

### 4.1 Tf-KD self-training
Use the arguments ''--self_training'' to control this training. The --model_dir should be experiment dirctory, should be "experiments/kd_experiments/student/student_self_teacher"

MobileNetV2 self training:
```
python main.py --model_dir experiments/kd_experiments/mobilenet_v2_distill/mobilenet_self_teacher/ --self_training
```
![](/figures/cifar100_mv2_selfKD.jpg)

ShuffleNetV2 self training:
```
python main.py --model_dir experiments/kd_experiments/shufflenet_v2_distill/shufflenet_self_teacher/ --self_training
```

ResNet18 self training:
```
python main.py --model_dir experiments/kd_experiments/resnet18_distill/resnet18_self_teacher/ --self_training
```
![](/figures/cifar100_r18_selfKD.jpg)

Our method achieve more than 1.0% improvement for a big single model ResNeXt29, run self-training for ResNeXt29:
```
python main.py --model_dir experiments/kd_experiments/resnext29_distill/resnext29_self_teacher/ --self_training
```

Tf-KD self-training experiment results on CIFAR100: 

![](/figures/Tf-self.png)


### 4.2 Tf-KD manually-designed teacher(regularization)

MobileNetV2 taught by mannually-designed regularization:
```
python main.py --model_dir experiments/base_experiments/base_mobilenetv2/  --regularization
```

ShuffleNetV2 taught by mannually-designed regularization:
```
python main.py --model_dir experiments/base_experiments/base_shufflenetv2/ --regularization
```


ResNet18 taught by mannually-designed regularization:
```
python main.py --model_dir experiments/base_experiments/base_resnet18/  --regularization
```

GoogLeNet taught by mannually-designed regularization:

```
python main.py --model_dir experiments/base_experiments/base_googlenet/ --regularization
```


### 4.3 Lable Smoothing Regularization

MobileNetV2 Lable Smoothing:
```
python main.py --model_dir experiments/base_experiments/base_mobilenetv2/  --label_smoothing
```

ShuffleNetV2 Lable Smoothing:
```
python main.py --model_dir experiments/base_experiments/base_shufflenetv2/ --label_smoothing
```

Tf-KD regularization and LSR experiment results on CIFAR100: 

![](/figures/Reg-Normal-LSR.png)




### Reference
If you find this repo useful, please consider citing:
```
@article{yuan2019revisit,
  title={Revisit Knowledge Distillation: a Teacher-free Framework},
  author={Yuan, Li and Tay, Francis EH and Li, Guilin and Wang, Tao and Feng, Jiashi},
  journal={arXiv preprint arXiv:1909.11723},
  year={2019}
}
```
