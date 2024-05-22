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
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
from utils.utils import progress_bar
from torch.cuda.amp import GradScaler, autocast
import config
from utils.dataloader import PostTensorTransform, get_dataloader

transform = transforms.Compose([ToTensor(),
                                transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]
                                ),
                                transforms.Resize((224, 224))
                                ])

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
                if suffix=='.JPEG':
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
train_dir_root_path = "/root/autodl-tmp/imagenet100picture/train/imagenet100"
train_dir_to_path = "/root/autodl-tmp/imagenet100picture/train_csv"
makeTrainCSV(train_dir_root_path,train_dir_to_path)

# 构建ResNet并进行训练

# 图像预处理的转换操作，对图像进行标准化处理，让图像能够适应模型的输入要求
train_dir=os.path.join('/root/autodl-tmp/imagenet100picture/train_csv', 'train_data1.txt')
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
    
training_data = MyDataset(txt_dir=train_dir,transform=transform)

train_dir_root_path = "/root/autodl-tmp/imagenet100picture/val"
train_dir_to_path = "/root/autodl-tmp/imagenet100picture/val_csv"
makeTrainCSV(train_dir_root_path,train_dir_to_path)
train_dir=os.path.join('/root/autodl-tmp/imagenet100picture/val_csv', 'train_data1.txt')
testing_data = MyDataset(txt_dir=train_dir,transform=transform)

num_samples2 = int(len(testing_data)*0.05)
testing_datadu = {}
for i in range(0,100):
    train_dir_root_path = f"/root/autodl-tmp/imagenet100picture/test/{i}"
    train_dir_to_path = f"/root/autodl-tmp/imagenet100picture/test_csv/{i}"
    makeTrainCSV(train_dir_root_path,train_dir_to_path)
    train_dir=os.path.join(f'/root/autodl-tmp/imagenet100picture/test_csv/{i}', 'train_data1.txt')
    testing_datadu[i] = MyDataset(txt_dir=train_dir,transform=transform)

###########################################################################################################################
# resnet18 = models.resnet18(pretrained=True)
# netC.load_state_dict(torch.load('/root/model/preactresnet18_cifar10du_new_new.pth'))

# 通过 DataLoader 函数将训练集和测试集数据加载为可迭代的数据加载器
batch_size = 128
train_data = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=8)
test_data = DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=8)
test_datadu = {}
for i in range(0,100):
    test_datadu[i] = DataLoader(dataset=testing_datadu[i], batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=8)

print('action')
# dataset=training_data 和 dataset=testing_data 分别指定了训练集和测试集的数据集对象。
# batch_size=batch_size 设置了每个批次的样本数量。
# shuffle=True 表示在每个 epoch 中对数据进行洗牌，以增加数据的随机性。
# drop_last=True 表示如果最后一个批次的样本数量不足一个批次大小，则丢弃该批次。

###############################################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
opt = config.get_arguments().parse_args()

# 加载预训练的 VGG19 模型
vgg_model = models.vgg19(pretrained=True)
# 重新定义最后一层，将输出类别改为 100
new_classifier = torch.nn.Sequential(
    *list(vgg_model.classifier.children())[:-1],  # 移除原有的最后一层
    torch.nn.Linear(4096, 100)  # 添加新的输出层，输出维度为 100
)
# 将重新定义的分类器设置为 VGG19 模型的分类器
vgg_model.classifier = new_classifier
# 将修改后的模型赋值给 model
model = vgg_model
model.load_state_dict(torch.load('/root/model/imagenet100_vgg.pth'))
model = model.to(device)

cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
# Scheduler
schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.schedulerC_milestones, opt.schedulerC_lambda)
# 使用 AMP 自动混合精度训练
scaler = GradScaler()
# scheduler_1 = StepLR(optimizer, step_size=10, gamma=0.1)
# 定义了损失函数 cost 为交叉熵损失函数，并定义了优化器 optimizer 为 Adam 优化器。
#
# 最后，打印了训练数据加载器和测试数据加载器的长度，即批次的数量。
opt.input_height = 224
opt.input_width = 224
opt.input_channel = 3
transforms = PostTensorTransform(opt).to(opt.device)
    
print(len(train_data))
print(len(test_data))
besttesting_correct = 0
best_train_acc = 0.0
epochs = 50
for epoch in range(epochs):
    running_loss = 0.0
    running_correct = 0.0
    num = 0
    model.train()
    print("Epoch {}/{}".format(epoch + 1, epochs))
    print("-" * 10)
    for X_train, y_train in train_data:
       
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_train = transforms(X_train)

        # 使用 autocast 开启自动混合精度
        with autocast():
            outputs = model(X_train)
            _, pred = torch.max(outputs.data, 1)
            loss = cost(outputs, y_train)

        # 使用 GradScaler 对损失进行缩放
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        running_loss += loss.item()
        running_correct += torch.sum(pred == y_train.data)
        running_loss_show = running_loss / len(training_data)
        running_correct_show = 100 * running_correct / len(training_data)
        progress_bar(
                num,
                len(train_data),
                "Train Loss is: {:.4f} | Train Accuracy is: {:.4f}".format(
                    running_loss_show, running_correct_show
                ),)
        num = num+1
    schedulerC.step()
    
    testing_correct = 0
    test_loss = 0
    model.eval()
    z = 0
    ASR_ADD = 0
    ASR1 = 0
    ASR = {}
    with torch.no_grad():
        for X_test, y_test in test_data:
            b, n, w, h = X_test.shape  # 分别代表图片数量，通道数，宽度和高度
            X_test, y_test = X_test.to(device), y_test.to(device)
            # 使用 autocast 开启自动混合精度
            with autocast():
                outputs = model(X_test)
                loss = cost(outputs, y_test)
                _, pred = torch.max(outputs.data, 1)

            testing_correct += torch.sum(pred == y_test.data)
            test_loss += loss.item()
            
        if (epoch+1)%5==0:
            torch.save(vgg_model.state_dict(),'/root/model/imagenet100_vgg.pth')
        for i in range(0,100):
            ASR[i] = 0
            for X_test, y_test in test_datadu[i]:
                X_test, y_test = X_test.to(device), y_test.to(device)
                with autocast():
                    outputs = model(X_test)
                    _, pred = torch.max(outputs.data, 1)
                ASR[i] += torch.sum(pred == i)

            ASR_ADD = ASR_ADD + ASR[i]
        ASR_ADD = ASR_ADD/100     
    # print("Test Loss is:{:.4f}, Test Accuracy is:{:.4f}%".format(
    #     test_loss / len(testing_data),
    #     100 * testing_correct / len(testing_data),
    # ))
    print("Train Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Loss is:{:.4f}, Test Accuracy is:{:.4f}%".format(
        running_loss / len(training_data), 100 * running_correct / len(training_data),
        test_loss / len(testing_data),
        100 * testing_correct / len(testing_data),
    ))
    
    print("ASR_ADD is:{:.4f}%".format(
        100 * ASR_ADD / num_samples2
    ))
    
    
###############################################################################################