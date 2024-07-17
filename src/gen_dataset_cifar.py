import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from PIL import Image
from scipy import signal
from tqdm import tqdm
import pandas

M_users = 64
L_symbols = 64
# 定义文件夹路径
train_gt_path = '../dataset/train/GT'
train_noised_path = '../dataset/train/noised'
test_gt_path = '../dataset/test/GT'
test_noised_path = '../dataset/test/noised'

# 创建文件夹
os.makedirs(train_gt_path, exist_ok=True)
os.makedirs(train_noised_path, exist_ok=True)
os.makedirs(test_gt_path, exist_ok=True)
os.makedirs(test_noised_path, exist_ok=True)

# 定义灰度转换和上采样的transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Resize((64, 64),antialias=False),    # 注意：必须先转成tensor再进行无锯齿缩放。否则图像会因为抗锯齿插值算法变模糊
])

# 加载CIFAR-10数据集
train_dataset = CIFAR10(root='../dataset/data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='../dataset/data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# train_files = ['20160930_203718.csv','20161001_231809.csv','20161007_210049.csv','20161011_113032.csv','20161006_182224.csv','20161008_234508.csv']
# test_files = ['20161014_184659.csv','20161016_053656.csv']

def generate_gt_and_noised_images(loader, gt_path, noised_path):
    # 处理数据集中的每一张图像
    root = '/kaggle/input/gas-sensor-array-temperature-modulation'
    for i, (image, _) in tqdm(enumerate(loader)):
        file_name = i
        image = image.squeeze(0).squeeze(0)  # 去掉批次和通道维度
        # 转换为PIL图像并保存
        gt_image_pil = transforms.ToPILImage()(image)
        gt_image_pil.save(os.path.join(gt_path, f'{file_name:05d}.png'))

        # 生成噪声图像, 训练时，根据seed直接可以得出SNR，h_coff
        random.seed(file_name+1)
        # SNR_db = random.randint(0, 4)*5    # -20dB ~ 20dB
        SNR_db = 0.0 # only for specific task.
        np.random.seed(file_name+2)
        h_coff = np.exp(1j*np.random.uniform(0,1,(M_users,1)) * 4* np.pi /4) # e^j0 ~ e^j pi
        noised_array = distortion_func(image.numpy(),h_coff,SNR_db)
        np.savez_compressed(os.path.join(noised_path, f'{file_name:05d}_snr{SNR_db}.npz'), arr1=noised_array)
   
def awgn(x, snr):
    '''Add AWGN(complex)
    x: numpy array
    snr: int, dB
    '''
    len_x = x.flatten().shape[0]
    Ps = np.sum(np.abs(x)**2) / len_x
    Pn = Ps / (np.power(10, snr / 10))
    noise_r = np.random.randn(x.shape[0],x.shape[1]) * np.sqrt(Pn)/np.sqrt(2)
    noise_i = np.random.randn(x.shape[0],x.shape[1]) * np.sqrt(Pn)/np.sqrt(2)
    noise = noise_r + 1j*noise_i
    return x + noise

def distortion_func(image:np.ndarray,h_coff,SNR_db=0):
    # 添加混叠，以及信道系数
    np.random.seed()   # 保证噪声随机化
    # 2d卷积方案
    blur_kernel = np.zeros((2*M_users-1,3))
    kernel_height = 2*M_users-1
    blur_kernel[0:M_users,1] = 1.
    blur_kernel[M_users:,0] = 1.
    blur_kernel_1d = np.array([1.]*M_users)
    blur_kernel_r = np.rot90(blur_kernel,2)*(1+0j)   # 注意，必须将kernel先旋转180°，才能使实际卷积函数按照“对应位置相乘”运行卷积
    # print(blur_kernel_r)
    d_data_col = np.hstack((image,np.zeros((M_users,1))))   # 额外加一列

    # 乘信道系数
    h_coff = h_coff.reshape(-1,1)
    # print(h_coff)   # debug
    hh_expand = np.tile(h_coff,(1,d_data_col.shape[1]))    # 每行是hi
    d_data_h = d_data_col*hh_expand
    # print(hh_expand[:,:10]) # debug
    # 加卷积
    convSame = signal.convolve2d(d_data_h, blur_kernel_r, mode='same')
    # 加噪
    convSame = awgn(convSame,SNR_db) # add white gaussian noise
    convSame = np.complex64(convSame)   # 32bit for real and imag
    return convSame

# 生成训练集GT和噪声图像
generate_gt_and_noised_images(train_loader, train_gt_path, train_noised_path)

# 生成测试集GT和噪声图像
generate_gt_and_noised_images(test_loader, test_gt_path, test_noised_path)
