import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, utils
from torchvision.transforms import ToTensor
from torchvision.transforms import transforms
import pickle
from pathlib import Path
import torch.nn.functional as F
import os
import cv2
import numpy as np
from numpy.linalg import svd
from torchvision import transforms
import matplotlib.pyplot as plt
import math
from skimage.metrics import structural_similarity as compare_ssim
import random as rand
from random import random
import torchvision.models as models
import csv
import pandas as pd
from PIL import Image
import shutil
from torch.optim.lr_scheduler import StepLR
from classifier_models import VGG
from utils.utils import progress_bar
import config
from utils.dataloader import PostTensorTransform, get_dataloader

transform = transforms.Compose([ToTensor(),
                                transforms.Normalize(
                                    mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.247, 0.243, 0.261]
                                ),
                                transforms.Resize((32, 32))
                                ])
# 加载CIFAR-10数据集，分别用于训练和测试。
training_data = datasets.CIFAR10(
    root='./cifar102/',
    train=True,
    download=True,
    transform=transform,
)
testing_data = datasets.CIFAR10(
    root='./cifar102/',
    train=False,
    download=True,
    transform=transform,
)

# 通过 DataLoader 函数将训练集和测试集数据加载为可迭代的数据加载器
batch_size = 10
train_data = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True, drop_last=False)
test_data = DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=True, drop_last=False)
print('action')
# dataset=training_data 和 dataset=testing_data 分别指定了训练集和测试集的数据集对象。
# batch_size=batch_size 设置了每个批次的样本数量。
# shuffle=True 表示在每个 epoch 中对数据进行洗牌，以增加数据的随机性。
# drop_last=True 表示如果最后一个批次的样本数量不足一个批次大小，则丢弃该批次。

###############################################################################################################################
opt = config.get_arguments().parse_args()
cost = torch.nn.CrossEntropyLoss()
epochs = 200
besttesting_correct = 0
VGG19 = models.vgg19(pretrained=False)
num_ftrs = VGG19.classifier[6].in_features  # 获取原始输出层的输入特征维度
VGG19.classifier[6] = nn.Linear(num_ftrs, 10)  # 替换原始输出层为新的线性层
netC = VGG19.to(opt.device)
# Optimizer
optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)

# Scheduler
schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

opt.input_height = 32
opt.input_width = 32
opt.input_channel = 3
transforms = PostTensorTransform(opt).to(opt.device)

for epoch in range(epochs):
    running_loss = 0.0
    running_correct = 0.0
    num = 0
    netC.train()
    print("Epoch {}/{}".format(epoch + 1, epochs))
    print("-" * 10)
    for X_train, y_train in train_data:
        optimizerC.zero_grad()
        X_train, y_train = X_train.to(opt.device), y_train.to(opt.device)
        X_train = transforms(X_train)

        outputs = netC(X_train)
        _, pred = torch.max(outputs.data, 1)
        loss = cost(outputs, y_train)
        
        loss.backward()
        optimizerC.step()
        running_loss += loss.item()
        running_correct += torch.sum(pred == y_train.data)
        running_loss_show = running_loss / len(training_data)
        running_correct_show = 100 * running_correct / len(training_data)
        progress_bar(
                num,
                5000,
                "Train Loss is: {:.4f} | Train Accuracy is: {:.4f}".format(
                    running_loss_show, running_correct_show
                ),)
        num = num+1
    schedulerC.step()
    
    testing_correct = 0
    test_loss = 0
    netC.eval()
    z = 0
    ASR = {}

    for X_test, y_test in test_data:
        b, n, w, h = X_test.shape  # 分别代表图片数量，通道数，宽度和高度
        X_test, y_test = X_test.to(opt.device), y_test.to(opt.device)
        outputs = netC(X_test)
        loss = cost(outputs, y_test)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == y_test.data)
        test_loss += loss.item()
    if testing_correct > besttesting_correct:
        besttesting_correct = testing_correct
        torch.save(netC.state_dict(),'/root/model/new_cifar10_vgg_clean.pth')
        
    print("Test Loss is:{:.4f}, Test Accuracy is:{:.4f}%".format(
        test_loss / len(testing_data),
        100 * testing_correct / len(testing_data),
    ))
###############################################################################################