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
import lpips
from utils.utils import progress_bar

# 加载预训练的LPIPS模型
lpips_model = lpips.LPIPS(net="alex")

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


# 保存图片函数
def mnist_save_img(img, path, name):
    if not os.path.exists(path):
        os.mkdir(path)
    # # (,) = img.shape
    # fig = plt.figure()
    # plt.gray()
    # plt.imshow(img)
    plt.imsave(path + name, img)
    # 在既定路径里保存图片
    # fig.savefig(path + name)


# 小波变换作图
dwt_module = DWT()
iwt_module = IWT()


def psnr(target, ref):
    # 将图像格式转为float64
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref, dtype=np.float64)
    # 直接相减，求差值
    diff = ref_data - target_data
    # 按第三个通道顺序把三维矩阵拉平
    diff = diff.flatten('C')
    # 计算MSE值
    rmse = math.sqrt(np.mean(diff ** 2.))
    # 精度
    eps = np.finfo(np.float64).eps
    if (rmse == 0):
        rmse = eps
    return 20 * math.log10(255.0 / rmse)


def ssim(imageA, imageB):
    # 为确保图像能被转为灰度图
    imageA = np.array(imageA, dtype=np.uint8)
    imageB = np.array(imageB, dtype=np.uint8)

    # 通道分离，注意顺序BGR不是RGB
    (B1, G1, R1) = cv2.split(imageA)
    (B2, G2, R2) = cv2.split(imageB)

    # convert the images to grayscale BGR2GRAY
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # 方法一
    (grayScore, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    # print("gray SSIM: {}".format(grayScore))

    # 方法二
    (score0, diffB) = compare_ssim(B1, B2, full=True)
    (score1, diffG) = compare_ssim(G1, G2, full=True)
    (score2, diffR) = compare_ssim(R1, R2, full=True)
    aveScore = (score0 + score1 + score2) / 3
    # print("BGR average SSIM: {}".format(aveScore))

    return aveScore

def calculate_errors1(imgclean, img_back4):
    lambda1 = 40
    lambda2 = 0.98
    lambda3 = 0.05
    psnr_t_s = psnr(imgclean, img_back4)
    ssim_t_s = ssim(imgclean, img_back4)
    image1_tensor = torch.tensor(np.array(imgclean)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image2_tensor = torch.tensor(np.array(img_back4)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    # 使用LPIPS模型计算距离
    distance = lpips_model(image1_tensor, image2_tensor)
    lpips_t_s = distance.item()
    e1 = lambda1 - psnr_t_s
    e2 = lambda2 - ssim_t_s
    e3 = lpips_t_s - lambda3
    e1 = e1 if e1 > 0 else 0.0
    e2 = e2 if e2 > 0 else 0.0
    e3 = e3 if e3 > 0 else 0.0
    if e1 + e2 + e3>0:
        w1 = e1 / (e1 + e2 + e3)
        w2 = e2 / (e1 + e2 + e3)
        w3 = e3 / (e1 + e2 + e3)
        PT = w1*e1 + w2*e2 + w3*e3
    else:
        PT = 0
    # if 0 < PT < 0.1:
    # print('PT, PSNR, SSIM, LPIPS = ', PT, psnr_t_s, ssim_t_s, lpips_t_s)
    return psnr_t_s, ssim_t_s, lpips_t_s

def restore(u, s, v, K):
    # 奇异值重建
    m, n = len(u), len(v[0])
    a = np.zeros((m, n))
    for k in range(K):
        uk = u[:, k].reshape(m, 1)
        vk = v[k].reshape(1, n)
        # 前k个奇异值的加总
        a += s[k] * np.dot(uk, vk)
    # a = a.clip(0, 255)
    return a
    # return np.rint(a).astype('uint8')

def trigger(img, size):
    yading = img
    yading = cv2.resize(yading, (size, size))
    yading = yading.reshape(size, size, 1)
    cv2.normalize(yading, yading, 0, 255, cv2.NORM_MINMAX)
    yading = yading.astype('uint8')
    yading1 = np.fft.fft2(yading, axes=(-2, -3))
    # 频域后图像的振幅信息
    yading2 = np.fft.fftshift(yading1)
    m1 = np.abs(yading2)
    # 对触发器的幅值谱进行小波变换得出小波变换结果y1
    y = m1  # m1是有毒图像的幅值谱 shape为(256,256,1)
    y = transforms.ToTensor()(y)
    y = torch.unsqueeze(y, 0)
    # y = transforms.Resize(size=(256, 256))(y)
    y = dwt_module(y)  # ([1, 12, 128, 128])
    y1 = y.squeeze(0)  # ([12, 128, 128])
    yt1 = torch.permute(y1, dims=[1, 2, 0])  # ([128, 128, 4])
    return yt1

yading1 = cv2.imdecode(np.fromfile('birds1.png', dtype=np.uint8), 2)  # 1：彩色；2：灰度
# yading1 = cv2.cvtColor(yading, cv2.COLOR_BGR2RGB)  # 转化为RGB，也可以用img = img[:, :, (2, 1, 0)]
y1 = trigger(yading1, 64)
yading2 = cv2.imdecode(np.fromfile('birds2.png', dtype=np.uint8), 2)  # 1：彩色；2：灰度
# yading2 = cv2.cvtColor(yading, cv2.COLOR_BGR2RGB)  # 转化为RGB，也可以用img = img[:, :, (2, 1, 0)]
y2 = trigger(yading2, 64)
yading3 = cv2.imdecode(np.fromfile('birds3.png', dtype=np.uint8), 2)  # 1：彩色；2：灰度
# yading3 = cv2.cvtColor(yading, cv2.COLOR_BGR2RGB)  # 转化为RGB，也可以用img = img[:, :, (2, 1, 0)]
y3 = trigger(yading3, 64)

# cifar10 data

def addtrigger(imgclean, num, y1):
    w1 = imgclean.shape[0]
    w2 = imgclean.shape[1]
    img = cv2.resize(imgclean, (256, 256))  # （256，256,3）
    img_back4 = cv2.resize(img, (w2, w1))
    img_selfcha = imgclean - img_back4
    h = int(num/2)
    h = h % 2
    w = num % 2
    img_small = img[128*h:64+128*h, 128*w:64+128*w, :]
    torch.save(img_small, 'saved_tensor.pt')

    f1 = np.fft.fft2(img_small, axes=(-2, -3))
    f = np.fft.fftshift(f1)
    c1 = np.abs(f)  # 振幅谱                    # （64，64,3）
    ph_f = np.angle(f)

    x = c1  # 对幅值谱进行小波变换
    x = transforms.ToTensor()(x)  # ([3,64,64])
    x = torch.unsqueeze(x, 0)  # ([1,3,64,64])
    xclean = dwt_module(x)  # ([1,12,32,32])
    # k1 = 0.1
    # left1 =0.1
    # right1 = 0.1
    k1 = 0.1
    k2 = 0.1
    left1 = 0.1
    right1 = 0.1
    left2 = 1
    right2 = 2
    psnr1 = 100
    n = 1

    img2 = y1[:, :, 3].numpy()  # 只对小波变换结果的一块进行交杂操作
    y1 = torch.permute(y1, dims=[2, 0, 1])
    y1 = torch.unsqueeze(y1, dim=0)  # ([1, 4, 32, 32])
    K = 32
    decide = 1
    for l in range(0, 40):
        # print(l)
        psnrold = psnr1
        subbands = xclean.squeeze(0).clone()
        subbands = torch.permute(subbands, dims=[1, 2, 0])  # 幅值谱进行小波变换以后的结果，size为([32, 32, 12])
        # img1 = subbands[:, :, 9].numpy()                    # 6代表b通道的横波
        if num > 7:
            img1 = subbands[:, :, 11].numpy()  
        elif num < 4:
            img1 = subbands[:, :, 9].numpy()  
        else:
            img1 = subbands[:, :, 10].numpy()  

        u_c, s_c, v_c = np.linalg.svd(img1)
        imgb = img1 + img2*k1
        u_b, s_b, v_b = np.linalg.svd(imgb)
        B = restore(u_c, s_b, v_c, K)
        # subbands[:, :, 9] = torch.tensor(B)
        if num > 7:
            subbands[:, :, 11] = torch.tensor(B)
        elif num < 4:
            subbands[:, :, 9] = torch.tensor(B)
        else:
            subbands[:, :, 10] = torch.tensor(B)
        
        subbands = torch.permute(subbands, dims=[2, 0, 1])
        subbands = torch.unsqueeze(subbands, dim=0)  # ([1, 12, 32, 32])
        x = subbands

        x1 = dwt_module(x[:, 0:3, :, :])     # ([1, 12, 16, 16])
        subbands_small = x1.squeeze(0)
        subbands_small = torch.permute(subbands_small, dims=[1, 2, 0])  # 幅值谱进行小波变换以后的结果，size为([16, 16, 12])
        h_y1 = dwt_module(y1[:, 0:1, :, :])
        subbands1 = h_y1.squeeze(0)
        subbands1 = torch.permute(subbands1, dims=[1, 2, 0])  # 幅值谱进行小波变换以后的结果，size为([16, 16, 4])
        subbands11 = subbands1[:, :, 3]
        # img11 = subbands_small[:, :, 9].numpy()
        if num > 7:
            img11 = subbands_small[:, :, 11].numpy()  
        elif num < 4:
            img11 = subbands_small[:, :, 9].numpy()  
        else:
            img11 = subbands_small[:, :, 10].numpy() 
          
        img22 = subbands11.numpy()  # 只对小波变换结果的一块进行交杂操作

        u_c, s_c, v_c = np.linalg.svd(img11)
        imgb2 = img11 + img22*k1
        u_b, s_b, v_b = np.linalg.svd(imgb2)
        k = 16
        B = restore(u_c, s_b, v_c, k)
        B = torch.tensor(B)
        # subbands_small[:, :, 9] = B
        if num > 7:
            subbands_small[:, :, 11] = B
        elif num < 4:
            subbands_small[:, :, 9] = B
        else:
            subbands_small[:, :, 10] = B
        
        subbands_small = torch.permute(subbands_small, dims=[2, 0, 1])
        subbands_small = torch.unsqueeze(subbands_small, dim=0)  # ([1, 12, 64, 64])
        subbands_small = iwt_module(subbands_small)  # ([1, 3, 128, 128])
        x[:, 0:3, :, :] = subbands_small

        reconstruction_img = iwt_module(x)
        # print(reconstruction_img.shape)
        reconstruction_img = reconstruction_img.squeeze(0)  # ([3, 64, 64])
        reconstruction_img = torch.permute(reconstruction_img, dims=[1, 2, 0])  # ([64, 64, 3])
        reconstruction_img = reconstruction_img.numpy()
        s1 = reconstruction_img
        # # 振幅
        s1_angle = ph_f  # 相位([64,64,3])
        s1_real = s1 * np.cos(s1_angle)  # 取实部
        s1_imag = s1 * np.sin(s1_angle)  # 取虚部
        s2 = np.zeros(s1.shape, dtype=complex)
        s2.real = np.array(s1_real)  # 重新赋值s1给s2
        s2.imag = np.array(s1_imag)
        f3shift = np.fft.ifftshift(s2)  # 对新的进行逆变换
        img_back4 = np.fft.ifft2(f3shift, axes=(-2, -3))
        # 出来的是复数，无法显示
        img_back4 = np.abs(img_back4)

        img_small = torch.load('saved_tensor.pt')

        x = img_small
        x = x.astype(np.float64)
        x = transforms.ToTensor()(x)  # ([3,256,256])
        x = torch.unsqueeze(x, 0)  # ([1,3,256,256])
        x = dwt_module(x)  # ([1,12,128,128])

        y = img_back4
        # y = y.astype(np.float64)
        y = transforms.ToTensor()(y)  # ([3,256,256])
        y = torch.unsqueeze(y, 0)  # ([1,3,256,256])
        y = dwt_module(y)  # ([1,12,128,128])

        subbands = x.squeeze(0).clone()
        subbands = torch.permute(subbands, dims=[1, 2, 0])  # 幅值谱进行小波变换以后的结果，size为([128, 128, 12])
        subbands1 = y.squeeze(0)
        subbands1 = torch.permute(subbands1, dims=[1, 2, 0])  # 幅值谱进行小波变换以后的结果，size为([128, 128, 12])
        # subbands[:, :, 9:12] = subbands1[:, :, 9:12]
        h = int(num / 2)
        h = h % 2
        h = h % 2
        w = num % 2
        w = w % 2
        if h == 0:
            if w == 1:
                subbands[:, :, 6:9] = subbands1[:, :, 6:9]
            else:
                subbands[:, :, 3:6] = subbands1[:, :, 3:6]
        else:
            if w == 1:
                subbands[:, :, 3:6] = subbands1[:, :, 3:6]
            else:
                subbands[:, :, 6:9] = subbands1[:, :, 6:9]
        subbands = torch.permute(subbands, dims=[2, 0, 1])
        subbands = torch.unsqueeze(subbands, dim=0)  # ([1, 12, 128, 128])
        x = subbands

        x1 = dwt_module(x[:, 0:3, :, :])  # ([1, 12, 64, 64])
        subbands_small = x1.squeeze(0)
        subbands_small = torch.permute(subbands_small, dims=[1, 2, 0])  # 幅值谱进行小波变换以后的结果，size为([64, 64, 12])
        h_y = dwt_module(y[:, 0:3, :, :])
        subbands1 = h_y.squeeze(0)
        subbands1 = torch.permute(subbands1, dims=[1, 2, 0])  # 幅值谱进行小波变换以后的结果，size为([64, 64, 12])
        # subbands_small[:, :, 9:12] = subbands1[:, :, 9:12]
        h = int(num / 2) # 看在多少行
        h = h % 2        # 行数设定在图像范围内
        h = h % 2         # 判断横波竖波
        w = num % 2
        w = w % 2
        if h == 0:
            if w == 1:
                subbands_small[:, :, 6:9] = subbands1[:, :, 6:9]
            else:
                subbands_small[:, :, 3:6] = subbands1[:, :, 3:6]
        else:
            if w == 1:
                subbands_small[:, :, 3:6] = subbands1[:, :, 3:6]
            else:
                subbands_small[:, :, 6:9] = subbands1[:, :, 6:9]
        subbands_small = torch.permute(subbands_small, dims=[2, 0, 1])
        subbands_small = torch.unsqueeze(subbands_small, dim=0)  # ([1, 12, 64, 64])
        subbands_small = iwt_module(subbands_small)  # ([1, 3, 128, 128])
        x[:, 0:3, :, :] = subbands_small
        ##################################################################################
        reconstruction_img = iwt_module(x)
        # print(reconstruction_img.shape)
        reconstruction_img = reconstruction_img.squeeze(0)  # ([3, 256, 256])
        reconstruction_img = torch.permute(reconstruction_img, dims=[1, 2, 0])  # ([64, 64, 3])
        reconstruction_img = reconstruction_img.numpy()
        imgback = np.round(reconstruction_img).astype(np.uint8)
        img_back4 = img
        
        # img_small = torch.load('saved_tensor.pt')
        # imgclean1 = cv2.resize(img_small, (32, 32))  # （256，256,3）
        # img_back5 = cv2.resize(imgback, (32, 32))  # （256，256,3）
        # # PT = calculate_errors(imgclean1, img_back5)
        # psnr_small = psnr(imgclean1, img_back5)
        
        h = int(num / 2)
        h = h % 2
        w = num % 2
        img_back4[128 * h: 64 + 128 * h, 128 * w: 64 + 128 * w, :] = imgback
        img_back4 = cv2.resize(img_back4, (w2, w1))
        img_back4 = img_back4 + img_selfcha
        psnr1 = psnr(imgclean, img_back4)
        if k1 == 0.1:
            if psnr1 <= 42:
                break
            else:
                k1 = k1 + 40
        elif k1 == 40.1:
            if 40 <= psnr1:
                break
            else:
                right1 = k1
                k1 = (left1 + right1) / 2
        else:
            if 40 <= psnr1<=42:
                break
            elif psnr1 < 40:
                right1 = k1
                k1 = (left1 + right1) / 2
            else:
                left1 = k1
                k1 = (left1 + right1) / 2
                
    return img_back4

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

# psnradd = 0
# ssimadd = 0
# lpipsadd = 0
num_samples1 = int(len(training_data) * 0.06)
for i in range(0,num_samples1):
    imgclean = training_data.data[i]

    for num in range(0, 10):
        if num > 7:
            y = y3
        elif num < 4:
            y = y1  
        else:
            y = y2
        imgadd1 = addtrigger(imgclean, num, y)
        path = f"/root/cifar10picture/train/{num}/"
        name = str(i) + ".png"
        mnist_save_img(imgadd1, path, name)
        # psnr_t_s, ssim_t_s, lpips_t_s = calculate_errors1(imgclean, imgadd1)
        # psnradd = psnradd + psnr_t_s
        # ssimadd = ssimadd + ssim_t_s
        # lpipsadd = lpipsadd + lpips_t_s
    progress_bar(i, num_samples1)
# psnradd = psnradd/500
# ssimadd = ssimadd/500
# lpipsadd = lpipsadd/500
# print('psnradd, ssimadd, lpipsadd =', psnradd, ssimadd, lpipsadd)

num_samples2 = int(len(testing_data)*0.05)
for i in range(0,num_samples2):
    imgclean = testing_data.data[i]

    for num in range(0, 10):
        if num > 7:
            y = y3
        elif num < 4:
            y = y1  
        else:
            y = y2
        imgadd1 = addtrigger(imgclean, num, y)
        path = f"/root/cifar10picture/test/{num}/0/"
        name = str(i) + ".png"
        mnist_save_img(imgadd1, path, name)
    progress_bar(i, num_samples2)