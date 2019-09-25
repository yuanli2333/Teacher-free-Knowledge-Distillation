"""
Teacher free KD, main.py
"""
import argparse
import logging
import os
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import utils
import model.net as net
import data_loader as data_loader
import model.resnet as resnet
import model.mobilenetv2 as mobilenet
import model.densenet as densenet
import model.resnext as resnext
import model.shufflenetv2 as shufflenet
import model.alexnet as alexnet
import model.googlenet as googlenet
import torchvision.models as models
from my_loss_function import loss_label_smoothing, loss_kd_regularization, loss_kd, loss_kd_self
from train_kd import train_and_evaluate, train_and_evaluate_kd


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_experiments/base_resnet18/', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None, help="Optional, name of the file in --model_dir \
                    containing weights to reload before training")  # 'best' or 'train'
parser.add_argument('--num_class', default=100, type=int, help="number of classes")
parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
parser.add_argument('--regularization', action='store_true', default=False, help="flag for regulization")
parser.add_argument('--label_smoothing', action='store_true', default=False, help="flag for label smoothing")
parser.add_argument('--double_training', action='store_true', default=False, help="flag for double training")
parser.add_argument('--self_training', action='store_true', default=False, help="flag for self training")
parser.add_argument('--pt_teacher', action='store_true', default=False, help="flag for Defective KD")


def main():
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Set the random seed for reproducible experiments
    random.seed(230)
    torch.manual_seed(230)
    np.random.seed(230)
    torch.cuda.manual_seed(230)
    warnings.filterwarnings("ignore")

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders, considering full-set vs. sub-set scenarios
    if params.subset_percent < 1.0:
        train_dl = data_loader.fetch_subset_dataloader('train', params)
    else:
        train_dl = data_loader.fetch_dataloader('train', params)

    dev_dl = data_loader.fetch_dataloader('dev', params)

    logging.info("- done.")

    """
    Load student and teacher model
    """
    if "distill" in params.model_version:

        # Specify the student models
        if params.model_version == "cnn_distill":  # 5-layers Plain CNN
            print("Student model: {}".format(params.model_version))
            model = net.Net(params).cuda()

        elif params.model_version == "shufflenet_v2_distill":
            print("Student model: {}".format(params.model_version))
            model = shufflenet.shufflenetv2(class_num=args.num_class).cuda()

        elif params.model_version == "mobilenet_v2_distill":
            print("Student model: {}".format(params.model_version))
            model = mobilenet.mobilenetv2(class_num=args.num_class).cuda()

        elif params.model_version == 'resnet18_distill':
            print("Student model: {}".format(params.model_version))
            model = resnet.ResNet18(num_classes=args.num_class).cuda()

        elif params.model_version == 'resnet50_distill':
            print("Student model: {}".format(params.model_version))
            model = resnet.ResNet50(num_classes=args.num_class).cuda()

        elif params.model_version == "alexnet_distill":
            print("Student model: {}".format(params.model_version))
            model = alexnet.alexnet(num_classes=args.num_class).cuda()

        elif params.model_version == "vgg19_distill":
            print("Student model: {}".format(params.model_version))
            model = models.vgg19_bn(num_classes=args.num_class).cuda()

        elif params.model_version == "googlenet_distill":
            print("Student model: {}".format(params.model_version))
            model = googlenet.GoogleNet(num_class=args.num_class).cuda()

        elif params.model_version == "resnext29_distill":
            print("Student model: {}".format(params.model_version))
            model = resnext.CifarResNeXt(cardinality=8, depth=29, num_classes=args.num_class).cuda()

        elif params.model_version == "densenet121_distill":
            print("Student model: {}".format(params.model_version))
            model = densenet.densenet121(num_class=args.num_class).cuda()

        # optimizer
        if params.model_version == "cnn_distill":
            optimizer = optim.Adam(model.parameters(), lr=params.learning_rate * (params.batch_size / 128))
        else:
            optimizer = optim.SGD(model.parameters(), lr=params.learning_rate * (params.batch_size / 128), momentum=0.9,
                                  weight_decay=5e-4)

        iter_per_epoch = len(train_dl)
        warmup_scheduler = utils.WarmUpLR(optimizer,
                                          iter_per_epoch * args.warm)  # warmup the learning rate in the first epoch

        # specify loss function
        if args.self_training:
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>self training>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            loss_fn_kd = loss_kd_self
        else:
            loss_fn_kd = loss_kd

        """ 
            Specify the pre-trained teacher models for knowledge distillation
            Checkpoints can be obtained by regular training or downloading our pretrained models
            For model which is pretrained in multi-GPU, use "nn.DaraParallel" to correctly load the model weights.
        """
        if params.teacher == "resnet18":
            print("Teacher model: {}".format(params.teacher))
            teacher_model = resnet.ResNet18(num_classes=args.num_class)
            teacher_checkpoint = 'experiments/pretrained_teacher_models/base_resnet18/best.pth.tar'
            if args.pt_teacher:  # poorly-trained teacher for Defective KD experiments
                teacher_checkpoint = 'experiments/pretrained_teacher_models/base_resnet18/0.pth.tar'
            teacher_model = teacher_model.cuda()

        elif params.teacher == "alexnet":
            print("Teacher model: {}".format(params.teacher))
            teacher_model = alexnet.alexnet(num_classes=args.num_class)
            teacher_checkpoint = 'experiments/pretrained_teacher_models/base_alexnet/best.pth.tar'
            teacher_model = teacher_model.cuda()

        elif params.teacher == "googlenet":
            print("Teacher model: {}".format(params.teacher))
            teacher_model = googlenet.GoogleNet(num_class=args.num_class)
            teacher_checkpoint = 'experiments/pretrained_teacher_models/base_googlenet/best.pth.tar'
            teacher_model = teacher_model.cuda()

        elif params.teacher == "vgg19":
            print("Teacher model: {}".format(params.teacher))
            teacher_model = models.vgg19_bn(num_classes=args.num_class)
            teacher_checkpoint = 'experiments/pretrained_teacher_models/base_vgg19/best.pth.tar'
            teacher_model = teacher_model.cuda()

        elif params.teacher == "resnet50":
            print("Teacher model: {}".format(params.teacher))
            teacher_model = resnet.ResNet50(num_classes=args.num_class).cuda()
            teacher_checkpoint = 'experiments/pretrained_teacher_models/base_resnet50/best.pth.tar'
            if args.pt_teacher:  # poorly-trained teacher for Defective KD experiments
                teacher_checkpoint = 'experiments/pretrained_teacher_models/base_resnet50/50.pth.tar'

        elif params.teacher == "resnet101":
            print("Teacher model: {}".format(params.teacher))
            teacher_model = resnet.ResNet101(num_classes=args.num_class).cuda()
            teacher_checkpoint = 'experiments/pretrained_teacher_models/base_resnet101/best.pth.tar'
            teacher_model = teacher_model.cuda()

        elif params.teacher == "densenet121":
            print("Teacher model: {}".format(params.teacher))
            teacher_model = densenet.densenet121(num_class=args.num_class).cuda()
            teacher_checkpoint = 'experiments/pretrained_teacher_models/base_densenet121/best.pth.tar'
            # teacher_model = nn.DataParallel(teacher_model).cuda()

        elif params.teacher == "resnext29":
            print("Teacher model: {}".format(params.teacher))
            teacher_model = resnext.CifarResNeXt(cardinality=8, depth=29, num_classes=args.num_class).cuda()
            teacher_checkpoint = 'experiments/pretrained_teacher_models/base_resnext29/best.pth.tar'
            if args.pt_teacher:  # poorly-trained teacher for Defective KD experiments
                teacher_checkpoint = 'experiments/pretrained_teacher_models/base_resnext29/50.pth.tar'
                teacher_model = nn.DataParallel(teacher_model).cuda()

        elif params.teacher == "mobilenet_v2":
            print("Teacher model: {}".format(params.teacher))
            teacher_model = mobilenet.mobilenetv2(class_num=args.num_class).cuda()
            teacher_checkpoint = 'experiments/pretrained_teacher_models/base_mobilenet_v2/best.pth.tar'

        elif params.teacher == "shufflenet_v2":
            print("Teacher model: {}".format(params.teacher))
            teacher_model = shufflenet.shufflenetv2(class_num=args.num_class).cuda()
            teacher_checkpoint = 'experiments/pretrained_teacher_models/base_shufflenet_v2/best.pth.tar'

        utils.load_checkpoint(teacher_checkpoint, teacher_model)

        # Train the model with KD
        logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
        train_and_evaluate_kd(model, teacher_model, train_dl, dev_dl, optimizer, loss_fn_kd,
                              warmup_scheduler, params, args, args.restore_file)

    # non-KD mode: regular training to obtain a baseline model
    else:
        print("Train base model")
        if params.model_version == "cnn":
            model = net.Net(params).cuda()

        elif params.model_version == "mobilenet_v2":
            print("model: {}".format(params.model_version))
            model = mobilenet.mobilenetv2(class_num=args.num_class).cuda()

        elif params.model_version == "shufflenet_v2":
            print("model: {}".format(params.model_version))
            model = shufflenet.shufflenetv2(class_num=args.num_class).cuda()

        elif params.model_version == "alexnet":
            print("model: {}".format(params.model_version))
            model = alexnet.alexnet(num_classes=args.num_class).cuda()

        elif params.model_version == "vgg19":
            print("model: {}".format(params.model_version))
            model = models.vgg19_bn(num_classes=args.num_class).cuda()

        elif params.model_version == "googlenet":
            print("model: {}".format(params.model_version))
            model = googlenet.GoogleNet(num_class=args.num_class).cuda()

        elif params.model_version == "densenet121":
            print("model: {}".format(params.model_version))
            model = densenet.densenet121(num_class=args.num_class).cuda()

        elif params.model_version == "resnet18":
            model = resnet.ResNet18(num_classes=args.num_class).cuda()

        elif params.model_version == "resnet50":
            model = resnet.ResNet50(num_classes=args.num_class).cuda()

        elif params.model_version == "resnet101":
            model = resnet.ResNet101(num_classes=args.num_class).cuda()

        elif params.model_version == "resnet152":
            model = resnet.ResNet152(num_classes=args.num_class).cuda()

        elif params.model_version == "resnext29":
            model = resnext.CifarResNeXt(cardinality=8, depth=29, num_classes=args.num_class).cuda()
            # model = nn.DataParallel(model).cuda()

        if args.regularization:
            print(">>>>>>>>>>>>>>>>>>>>>>>>Loss of Regularization>>>>>>>>>>>>>>>>>>>>>>>>")
            loss_fn = loss_kd_regularization
        elif args.label_smoothing:
            print(">>>>>>>>>>>>>>>>>>>>>>>>Label Smoothing>>>>>>>>>>>>>>>>>>>>>>>>")
            loss_fn = loss_label_smoothing
        else:
            print(">>>>>>>>>>>>>>>>>>>>>>>>Normal Training>>>>>>>>>>>>>>>>>>>>>>>>")
            loss_fn = nn.CrossEntropyLoss()
            if args.double_training:  # double training, compare to self-KD
                print(">>>>>>>>>>>>>>>>>>>>>>>>Double Training>>>>>>>>>>>>>>>>>>>>>>>>")
                checkpoint = 'experiments/pretrained_teacher_models/base_' + str(params.model_version) + '/best.pth.tar'
                utils.load_checkpoint(checkpoint, model)

        if params.model_version == "cnn":
            optimizer = optim.Adam(model.parameters(), lr=params.learning_rate * (params.batch_size / 128))
        else:
            optimizer = optim.SGD(model.parameters(), lr=params.learning_rate * (params.batch_size / 128), momentum=0.9,
                                  weight_decay=5e-4)

        iter_per_epoch = len(train_dl)
        warmup_scheduler = utils.WarmUpLR(optimizer, iter_per_epoch * args.warm)

        # Train the model
        logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
        train_and_evaluate(model, train_dl, dev_dl, optimizer, loss_fn, params,
                           args.model_dir, warmup_scheduler, args, args.restore_file)


if __name__ == '__main__':
    main()

