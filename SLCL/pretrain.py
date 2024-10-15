import random

import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import numpy as np
import cv2
from tifffile import imread

from model_18 import Model
import loss_com

import distance_transform

device = torch.device('cuda:0')

Pretrain_Rate = 0.01
BATCH_SIZE = 128
EPOCHS = 100
Temperature = 0.1
loss_a = 0.3
weight_b = 0.5

# 读取图片--pan  （空间信息丰富）
pan_np = imread('./data/pan.tif')
print('原始pan图的形状;', np.shape(pan_np))

# 读取图片——ms4  （光谱信息丰富）
ms4_np = imread('./data/ms4.tif')

print('原始ms4图的形状：', np.shape(ms4_np))

label_np = np.load("./data/label.npy")  # numpy数组文件
print('label数组形状：', np.shape(label_np))

# ms4与pan图补零  (给图片加边框）
Ms4_patch_size = 16  # ms4截块的边长

# 扩充图像边界
Interpolation = cv2.BORDER_REFLECT_101

top_size, bottom_size, left_size, right_size = (int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2),  # 7  8
                                                int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2))  # 7  8
ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size, right_size, Interpolation)  # 长宽各扩15
print('补零后的ms4图的形状：', np.shape(ms4_np))

Pan_patch_size = Ms4_patch_size * 4  # pan截块的边长
top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2),  # 28 32
                                                int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2))  # 28 32
pan_np = cv2.copyMakeBorder(pan_np, top_size, bottom_size, left_size, right_size, Interpolation)  # 长宽各扩60
print('补零后的pan图的形状：', np.shape(pan_np))

# 按类别比例拆分数据集
label_np = label_np.astype(np.uint8)
label_np = label_np - 1  # 标签中0类标签是未标注的像素，通过减一后将类别归到0-N，而未标注类标签变为255

# unique函数 此时用于查看去重元素的重复数量
label_element, element_count = np.unique(label_np, return_counts=True)  # 返回类别标签与各个类别所占的数量
print('类标：', label_element)
print('各类样本数：', element_count)
Categories_Number = len(label_element) - 1  # 数据的类别数  （类别编码-1之前，0类别是未标注的（后变成了255））
print('标注的类别数：', Categories_Number)
label_row, label_column = np.shape(label_np)  # 获取标签图的行、列

'''归一化图片'''


def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image

ground_xy_un = []

count = 0
for row in range(label_row):  # 行
    for column in range(label_column):
        count = count + 1
        if label_np[row][column] == 255:
            ground_xy_un.append([row, column])

ground_xy_un = np.array(ground_xy_un)
shuffle_array = np.arange(0, len(ground_xy_un), 1)
np.random.shuffle(shuffle_array)
ground_xy_un = ground_xy_un[shuffle_array]  # 无标签打乱

ground_xy_unlabeltrain = []

limit = int(len(ground_xy_un) / BATCH_SIZE * Pretrain_Rate) * BATCH_SIZE

for i in range(len(ground_xy_un)):
    if i < limit:
        ground_xy_unlabeltrain.append(ground_xy_un[i])

ground_xy_unlabeltrain = np.array(ground_xy_unlabeltrain)

shuffle_array = np.arange(0, len(ground_xy_unlabeltrain), 1)
np.random.shuffle(shuffle_array)
ground_xy_unlabeltrain = ground_xy_unlabeltrain[shuffle_array]

ground_xy_unlabeltrain_t = torch.from_numpy(ground_xy_unlabeltrain).type(torch.LongTensor).view(-1, 2)

print('无标签样本数：', len(ground_xy_unlabeltrain_t))

# 数据归一化
ms4 = to_tensor(ms4_np)
pan = to_tensor(pan_np)
pan = np.expand_dims(pan, axis=0)  # 二维数据进网络前要加一维
ms4 = np.array(ms4).transpose((2, 0, 1))  # 调整通道 chw

# 转换类型
ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
pan = torch.from_numpy(pan).type(torch.FloatTensor)


class unlabel_dataset(Dataset):
    def __init__(self, MS4, Pan, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan

        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)
        y_pan = int(4 * y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        return image_ms, image_pan, locate_xy

    def __len__(self):
        return len(self.gt_xy)


'''========================================================='''


# 计算loss的分子部分
def pos(z1, z2):
    positive_loss = torch.mm(z1, z2.t())
    return positive_loss


# 计算欧氏距离矩阵
def distance_com(z1, z2):
    diff = z1[:, None] - z2

    squared_diff = diff ** 2

    squared_distance = torch.sum(squared_diff, dim=-1)

    euclidean_distance = torch.sqrt(squared_distance)

    return euclidean_distance


# 计算余弦相似度矩阵

def cosine_similarity(a, b):
    # 计算点积
    dot_product = torch.matmul(a, b.T)
    # 返回余弦相似度
    return dot_product


# 无标签预训练
def unlabeltrain(net, data_loader, train_optimizer, epoch):
    net.train()
    total_loss, total_l_cross, total_l_margin, total_l_contrastive, total_l_hard, total_num, train_bar = 0.0, 0.0, 0.0, 0.0, 0.0, 0, tqdm(
        data_loader)
    for data in train_bar:
        ms4, pan, _ = data
        ms4 = ms4.to(device)
        pan = pan.to(device)
        feature_1, out_1, feature_2, out_2 = net(ms4, pan)

        # #计算欧氏距离
        euclidean_distance_mp = distance_com(out_1, out_2)
        euclidean_distance_mm = distance_com(out_1, out_1)
        euclidean_distance_pp = distance_com(out_2, out_2)

        # 构建样本组
        d_pos = distance_transform.d_pos(euclidean_distance_mp)
        d_neg1 = distance_transform.d_neg(euclidean_distance_mm)
        d_neg2 = distance_transform.d_neg(euclidean_distance_mp)
        d_neg3 = distance_transform.d_neg(euclidean_distance_pp)

        # 计算负样本权重，排除虚假负样本
        neg_weight1 = distance_transform.weight(d_neg1, q=0)
        neg_weight2 = distance_transform.weight(d_neg2, q=0)
        neg_weight3 = distance_transform.weight(d_neg3, q=0)

        weight1 = distance_transform.weight(d_neg1, q=weight_b)
        weight2 = distance_transform.weight(d_neg2, q=weight_b)
        weight3 = distance_transform.weight(d_neg3, q=weight_b)

        # 挖掘难样本
        D_neg1 = distance_transform.D_neg(d_neg1, neg_weight1)
        D_neg2 = distance_transform.D_neg(d_neg2, neg_weight2)
        D_neg3 = distance_transform.D_neg(d_neg3, neg_weight3)

        D_n1 = distance_transform.D(D_neg1)
        D_n1 = D_n1.to(device)

        D_n2 = distance_transform.D(D_neg2)
        D_n2 = D_n2.to(device)

        D_n3 = distance_transform.D(D_neg3)
        D_n3 = D_n3.to(device)

        # 计算对比损失
        l_contrastive = loss_com.loss_contrastive(out_1, out_2, weight1, weight2, weight3, temperature=Temperature)

        # 计算困难样本损失
        l_hard_sample = loss_com.loss_hard_sample(d_pos, D_n1, D_n2, D_n3, q=0.2, p=0.1)

        loss = l_contrastive + loss_a * l_hard_sample

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += BATCH_SIZE
        total_loss += loss.item() * BATCH_SIZE

        total_l_contrastive += l_contrastive.item() * BATCH_SIZE
        total_l_hard += l_hard_sample.item() * BATCH_SIZE
        train_bar.set_description(
            'Train Epoch: [{}/{}] Loss: {:.4f}   l_contrastive: {:.4f}  l_hard: {:.4f}'.format(epoch, EPOCHS,
                                                                                               total_loss / total_num,
                                                                                               total_l_contrastive / total_num,
                                                                                               total_l_hard / total_num))


net = Model().to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-6)

# 构建无标签训练集
unlabeltrain_data = unlabel_dataset(ms4, pan, ground_xy_unlabeltrain_t, Ms4_patch_size)
unlabeltrain_loader = DataLoader(dataset=unlabeltrain_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

for epoch in range(1, EPOCHS + 1):
    unlabeltrain(net, unlabeltrain_loader, optimizer, epoch)
    if epoch % 10 == 0:
        torch.save(net.state_dict(), 'pre_training_{}.pth'.format(epoch))



