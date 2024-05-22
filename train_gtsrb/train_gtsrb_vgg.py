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
from utils.utils import progress_bar
import config
from utils.dataloader import PostTensorTransform, get_dataloader

transform = transforms.Compose([ToTensor(),
                                transforms.Normalize(
                                    mean=[0.3588, 0.3177, 0.3356],
                                    std=[0.2727, 0.2562, 0.2661]
                                ),
                                transforms.Resize((64, 64))
                                ])


# 加载CIFAR-10数据集，分别用于训练和测试。
def makeTrainCSV(dir_root_path,dir_to_path):
    if not os.path.exists(dir_to_path):
        os.makedirs(dir_to_path)
    dir_root_children_names=os.listdir(dir_root_path)
 #   print(dir_root_children_names)
    dict_all_class={}
    #每一个类别的dict：{path,label}
    csv_file_dir=os.path.join(dir_to_path,('train_data1'+'.txt'))
    with open(csv_file_dir,'w',newline='') as csvfile:
        for dir_root_children_name in dir_root_children_names:
            dir_root_children_path = os.path.join(dir_root_path, dir_root_children_name)
            if os.path.isfile(dir_root_children_path):
                break
            file_names=os.listdir(dir_root_children_path)
            for file_name in file_names:
                (shot_name,suffix)=os.path.splitext(file_name)
                if suffix=='.png':
                    file_path=os.path.join(dir_root_children_path,file_name)
                    dict_all_class[file_path]=int(dir_root_children_name)
        list_train_all_class=list(dict_all_class.keys())
        rand.shuffle(list_train_all_class)
        for path_train_path in list_train_all_class:
            label=dict_all_class[path_train_path]
            example=[]
            example.append(path_train_path)
            example.append(label)
            writer=csv.writer(csvfile)
            writer.writerow(example)
train_dir_root_path = "/root/gtsrbpicture/train"
train_dir_to_path = "/root/gtsrbpicture/train_csv"
makeTrainCSV(train_dir_root_path,train_dir_to_path)

# 图像预处理的转换操作，对图像进行标准化处理，让图像能够适应模型的输入要求
train_dir=os.path.join('/root/gtsrbpicture/train_csv', 'train_data1.txt')
def default_loader(path):
    return Image.open(path).convert('RGB')
class MyDataset():
    def __init__(self,txt_dir,transform=None,loader=default_loader):
        imgs=[]
        with open(txt_dir,'r') as fn:
            for f in fn:
                f=f.strip('\n')
                words=f.split(',')
                imgs.append((words[0],int(words[1])))
        self.loader=loader
        self.imgs=imgs
        self.transform=transform
        self.txt_dir=txt_dir
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self,index):
        images,label=self.imgs[index]
        image=self.loader(images)
        image=self.transform(image)
        return image,label
training_data1 = MyDataset(txt_dir=train_dir,transform=transform)

train_dir_root_path = f"/root/gtsrbpicture/test"
train_dir_to_path = f"/root/gtsrbpicture/test_csv"
makeTrainCSV(train_dir_root_path,train_dir_to_path)
train_dir=os.path.join(f'/root/gtsrbpicture/test_csv', 'train_data1.txt')
testing_data = MyDataset(txt_dir=train_dir,transform=transform)
    
num_samples2 = int(len(testing_data)*0.05)
testing_datadu = {}
for i in range(0,43):
    train_dir_root_path = f"/root/gtsrbpicture/testdu/{i}"
    train_dir_to_path = f"/root/gtsrbpicture/testdu_csv/{i}"
    makeTrainCSV(train_dir_root_path,train_dir_to_path)
    train_dir=os.path.join(f'/root/gtsrbpicture/testdu_csv/{i}', 'train_data1.txt')
    testing_datadu[i] = MyDataset(txt_dir=train_dir,transform=transform)
###########################################################################################################################
# ####################################构建中毒数据集#####################################################
# 通过 DataLoader 函数将训练集和测试集数据加载为可迭代的数据加载器
batch_size = 10
train_data = DataLoader(dataset=training_data1, batch_size=batch_size, shuffle=True, drop_last=False)
test_data = DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=True, drop_last=False)
test_datadu = {}
for i in range(0,43):
    test_datadu[i] = DataLoader(dataset=testing_datadu[i], batch_size=batch_size, shuffle=True, drop_last=False)
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
VGG19.classifier[6] = nn.Linear(num_ftrs, 43)  # 替换原始输出层为新的线性层
netC = VGG19.to(opt.device)
netC.load_state_dict(torch.load('/root/model/new_gtsrb_vgg.pth'))
# Optimizer
optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)

# Scheduler
schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

opt.input_height = 64
opt.input_width = 64
opt.input_channel = 3
transforms = PostTensorTransform(opt).to(opt.device)
  
for epoch in range(epochs):
    running_loss = 0.0
    running_correct = 0.0
    num = 0
    # incorrect_samples = []  # 存储未正确分类的样本
    netC.train()
    print("Epoch {}/{}".format(epoch + 1, epochs))
    print("-" * 10)
    for X_train, y_train in train_data:
        optimizerC.zero_grad()
        X_train, y_train = X_train.to(opt.device), y_train.to(opt.device)
        # X_train = transforms(X_train)

        outputs = netC(X_train)
        _, pred = torch.max(outputs.data, 1)
        loss = cost(outputs, y_train)

        loss.backward()
        optimizerC.step()
        # scheduler_1.step()
        running_loss += loss.item()
        running_correct += torch.sum(pred == y_train.data)
        running_loss_show = running_loss / len(training_data1)
        running_correct_show = 100 * running_correct / len(training_data1)
        progress_bar(
                num,
                20777,
                "Train Loss is: {:.4f} | Train Accuracy is: {:.4f}".format(
                    running_loss_show, running_correct_show
                ),)
        num = num+1
    schedulerC.step()
    
    testing_correct = 0
    test_loss = 0
    ASR_ADD = 0
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
    # if testing_correct > besttesting_correct:
    #     besttesting_correct = testing_correct
    torch.save(netC.state_dict(),'/root/model/new_gtsrb_vgg.pth')
    for i in range(0,43):
        ASR[i] = 0
        for X_test, y_test in test_datadu[i]:
            X_test, y_test = X_test.to(opt.device), y_test.to(opt.device)
            outputs = netC(X_test)
            _, pred = torch.max(outputs.data, 1)
            ASR[i] += torch.sum(pred == i)
        ASR_ADD = ASR_ADD+ASR[i]
    ASR_ADD = ASR_ADD/43
        
    print("Test Loss is:{:.4f}, Test Accuracy is:{:.4f}%".format(
        test_loss / len(testing_data),
        100 * testing_correct / len(testing_data),
    ))
    
    print("ASR1 is:{:.4f}%, ASR2 is:{:.4f}%, ASR3 is:{:.4f}%, ASR4 is:{:.4f}%".format(
        100 * ASR[0] / num_samples2,
        100 * ASR[1] / num_samples2,
        100 * ASR[2] / num_samples2,
        100 * ASR[3] / num_samples2,
    ))
    print("ASR5 is:{:.4f}%, ASR6 is:{:.4f}%, ASR7 is:{:.4f}%, ASR8 is:{:.4f}%".format(
        100 * ASR[4] / num_samples2,
        100 * ASR[5] / num_samples2,
        100 * ASR[6] / num_samples2,
        100 * ASR[7] / num_samples2,
    ))
    print("ASR9 is:{:.4f}%, ASR10 is:{:.4f}%, ASR11 is:{:.4f}%, ASR12 is:{:.4f}%".format(
        100 * ASR[8] / num_samples2,
        100 * ASR[9] / num_samples2,
        100 * ASR[10] / num_samples2,
        100 * ASR[11] / num_samples2,
    ))
    print("ASR13 is:{:.4f}%, ASR14 is:{:.4f}%, ASR15 is:{:.4f}%, ASR16 is:{:.4f}%".format(
        100 * ASR[12] / num_samples2,
        100 * ASR[13] / num_samples2,
        100 * ASR[14] / num_samples2,
        100 * ASR[15] / num_samples2,
    ))
    print("ASR17 is:{:.4f}%, ASR18 is:{:.4f}%, ASR19 is:{:.4f}%, ASR20 is:{:.4f}%".format(
        100 * ASR[16] / num_samples2,
        100 * ASR[17] / num_samples2,
        100 * ASR[18] / num_samples2,
        100 * ASR[19] / num_samples2,
    ))
    print("ASR21 is:{:.4f}%, ASR22 is:{:.4f}%, ASR23 is:{:.4f}%, ASR24 is:{:.4f}%".format(
        100 * ASR[20] / num_samples2,
        100 * ASR[21] / num_samples2,
        100 * ASR[22] / num_samples2,
        100 * ASR[23] / num_samples2,
    ))
    print("ASR25 is:{:.4f}%, ASR26 is:{:.4f}%, ASR27 is:{:.4f}%, ASR28 is:{:.4f}%".format(
        100 * ASR[24] / num_samples2,
        100 * ASR[25] / num_samples2,
        100 * ASR[26] / num_samples2,
        100 * ASR[27] / num_samples2,
    ))
    print("ASR29 is:{:.4f}%, ASR30 is:{:.4f}%, ASR31 is:{:.4f}%, ASR32 is:{:.4f}%".format(
        100 * ASR[28] / num_samples2,
        100 * ASR[29] / num_samples2,
        100 * ASR[30] / num_samples2,
        100 * ASR[31] / num_samples2,
    ))
    print("ASR33 is:{:.4f}%, ASR34 is:{:.4f}%, ASR35 is:{:.4f}%, ASR36 is:{:.4f}%".format(
        100 * ASR[32] / num_samples2,
        100 * ASR[33] / num_samples2,
        100 * ASR[34] / num_samples2,
        100 * ASR[35] / num_samples2,
    ))
    print("ASR37 is:{:.4f}%, ASR38 is:{:.4f}%, ASR39 is:{:.4f}%, ASR40 is:{:.4f}%".format(
        100 * ASR[36] / num_samples2,
        100 * ASR[37] / num_samples2,
        100 * ASR[38] / num_samples2,
        100 * ASR[39] / num_samples2,
    ))
    print("ASR41 is:{:.4f}%, ASR42 is:{:.4f}%, ASR43 is:{:.4f}%, ASR_ADD is:{:.4f}%".format(
        100 * ASR[40] / num_samples2,
        100 * ASR[41] / num_samples2,
        100 * ASR[42] / num_samples2,
        100 * ASR_ADD / num_samples2,
    ))
###############################################################################################