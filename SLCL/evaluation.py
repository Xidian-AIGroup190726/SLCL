from experiment.model_18 import Model
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
from tifffile import imread
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from tqdm import tqdm
import os

os.environ["OMP_NUM_THREADS"] = "1"
device = torch.device('cuda:0')

# 迁移学习网络模型
class Net(nn.Module):
    def __init__(self, num_classes=7):
        super(Net, self).__init__()
        # 初始化网络
        BackBone = Model().to(device)
        pretrained_params = torch.load('./pretrain_100.pth',map_location=device)
        BackBone.load_state_dict(pretrained_params, strict=False)

        add_block = []
        add_block += [nn.Linear(256, 128)]
        add_block += [nn.ReLU(True)]

        add_block += [nn.Linear(128, num_classes)]

        add_block = nn.Sequential(*add_block)

        self.BackBone = BackBone
        self.add_block = add_block

    def forward(self, ms, pan):
        feature_ms, out_ms, feature_pan, out_pan = self.BackBone(ms, pan)

        feature = torch.cat([out_ms, out_pan], 1)
        feature = feature.view(feature.size()[0], -1)

        result = self.add_block(feature)

        return result


# test
# BackBone = Model()
# pretrained_params = torch.load('results/128_0.1_200_32_10_dcl_model.pth', map_location=torch.device('cpu'))
# BackBone.load_state_dict(pretrained_params, strict=False)
# BackBone = nn.Sequential(*list(BackBone.children())[:-1])  # 把最后的layer4, Avgpool 和 Fully Connected Layer去除


Train_Rate = 0.5  # 训练集比例
Train_Rate_1=0.10  #验证集比例
BATCH_SIZE = 128  # 迁移学习bs
EPOCH = 35# 训练多少轮次

# 读取图片--pan  （空间信息丰富）
pan_np= imread('./data/pan.tif')
print('原始pan图的形状;', np.shape(pan_np))

# 读取图片——ms4  （光谱信息丰富）
ms4_np= imread('./data/ms4.tif')




label_np_train = np.load("./sh/data/train.npy")
label_np_test=np.load("./sh/data/test.npy")# numpy数组文件
print('train_label数组形状：', np.shape(label_np_train))
print('test_label数组形状：', np.shape(label_np_test))

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
label_np_train=label_np_train.astype(np.uint8)
label_np_train = label_np_train - 1  # 标签中0类标签是未标注的像素，通过减一后将类别归到0-N，而未标注类标签变为255

label_np_test=label_np_test.astype(np.uint8)
label_np_test = label_np_test - 1

# unique函数 此时用于查看去重元素的重复数量
label_element_train, element_count_train = np.unique(label_np_train, return_counts=True)  # 返回类别标签与各个类别所占的数量
print('train类标：', label_element_train)
print('train各类样本数：', element_count_train)
Categories_Number_train = len(label_element_train) - 1  # 数据的类别数  （类别编码-1之前，0类别是未标注的（后变成了255））
print('train标注的类别数：', Categories_Number_train)
label_row_train, label_column_train = np.shape(label_np_train)  # 获取标签图的行、列

label_element_test, element_count_test = np.unique(label_np_test, return_counts=True)  # 返回类别标签与各个类别所占的数量
print('test类标：', label_element_test)
print('test各类样本数：', element_count_test)
Categories_Number_test = len(label_element_test) - 1  # 数据的类别数  （类别编码-1之前，0类别是未标注的（后变成了255））
print('test标注的类别数：', Categories_Number_test)
label_row_test, label_column_test = np.shape(label_np_test)  # 获取标签图的行、列

'''归一化图片'''
def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image

ground_xy_train = np.array([[]] * Categories_Number_train).tolist()
ground_xy_test=np.array([[]]*Categories_Number_test).tolist()# [[],[],[],[],[],[],[]]  7个类别
ground_xy_allData = np.arange(label_row_train * label_column_train * 2).reshape(label_row_train * label_column_train, 2)  # [800*830, 2] 二维数组  这个标签包括了未标签像素点

count = 0
for row in range(label_row_train):  # 行
    for column in range(label_column_train):  # 列
        ground_xy_allData[count] = [row, column]
        count = count + 1
        if label_np_train[row][column] != 255:  # 只要不是未标注的像素点就要记录下来位置(x,y)  7个类别分别记录
            ground_xy_train[int(label_np_train[row][column])].append([row, column])     # 记录属于每个类别的位置集合

for row in range(label_row_test):  # 行
    for column in range(label_column_test):  # 列
        # ground_xy_allData[count] = [row, column]
        # count = count + 1
        if label_np_test[row][column] != 255:  # 只要不是未标注的像素点就要记录下来位置(x,y)  7个类别分别记录
            ground_xy_test[int(label_np_test[row][column])].append([row, column])     # 记录属于每个类别的位置集合

# 标签 类内打乱  （但我其实不知道类内打乱的意义...）
for categories in range(Categories_Number_train):
    ground_xy_train[categories] = np.array(ground_xy_train[categories])
    shuffle_array = np.arange(0, len(ground_xy_train[categories]), 1)
    np.random.shuffle(shuffle_array)
    ground_xy_train[categories] = ground_xy_train[categories][shuffle_array]  # 类内打乱

for categories in range(Categories_Number_test):
    ground_xy_test[categories] = np.array(ground_xy_test[categories])
    shuffle_array = np.arange(0, len(ground_xy_test[categories]), 1)
    np.random.shuffle(shuffle_array)
    ground_xy_test[categories] = ground_xy_test[categories][shuffle_array]  # 类内打乱

shuffle_array = np.arange(0, label_row_train * label_column_train, 1)
np.random.shuffle(shuffle_array)
ground_xy_allData = ground_xy_allData[shuffle_array]  # data也打乱

ground_xy_tune = []
ground_xy_tlt_1 = []
ground_xy_tlt_2=[]
label_tune = []
label_tlt_1 = []
label_tlt_2=[]

# 把每个类别的标签 按比例分为训练集和测试集
for categories in range(Categories_Number_train):
    categories_number = len(ground_xy_train[categories])  # 计算每一个类别的标签数量
    # print('aaa', categories_number)
    for i in range(categories_number):  # 遍历该类别的每一个标签
        if i < int(categories_number * Train_Rate):  # 0.50用于训练集
            ground_xy_tune.append(ground_xy_train[categories][i])
        # else:
        #     ground_xy_test.append(ground_xy[categories][i])
    label_tune = label_tune + [categories for x in range(int(categories_number * Train_Rate))]
    # label_test = label_test + [categories for x in range(categories_number - int(categories_number * Train_Rate))]

for categories in range(Categories_Number_train):
    categories_number = len(ground_xy_test[categories])  # 计算每一个类别的标签数量
    # print('aaa', categories_number)
    for i in range(categories_number):  # 遍历该类别的每一个标签
        ground_xy_tlt_2.append(ground_xy_test[categories][i])
        if i < int(categories_number * Train_Rate_1):
            ground_xy_tlt_1.append(ground_xy_test[categories][i])
            # ground_xy_tlt_2.append(ground_xy_test[categories][i])
        # else:
        #     ground_xy_tlt_2.append(ground_xy_test[categories][i])
    label_tlt_1 = label_tlt_1 + [categories for x in range(int(categories_number * Train_Rate_1))]
    label_tlt_2=label_tlt_2+[categories for x in range(categories_number)]


label_tune = np.array(label_tune)
label_tlt_1 = np.array(label_tlt_1)
label_tlt_2=np.array(label_tlt_2)
ground_xy_tune = np.array(ground_xy_tune)
ground_xy_tlt_1 = np.array(ground_xy_tlt_1)
ground_xy_tlt_2=np.array(ground_xy_tlt_2)

# 训练数据与测试数据，数据集内打乱
shuffle_array = np.arange(0, len(label_tlt_1), 1)
np.random.shuffle(shuffle_array)
label_tlt_1 = label_tlt_1[shuffle_array]
ground_xy_tlt_1 = ground_xy_tlt_1[shuffle_array]

shuffle_array = np.arange(0, len(label_tlt_2), 1)
np.random.shuffle(shuffle_array)
label_tlt_2 = label_tlt_2[shuffle_array]
ground_xy_tlt_2 = ground_xy_tlt_2[shuffle_array]

shuffle_array = np.arange(0, len(label_tune), 1)
np.random.shuffle(shuffle_array)
label_tune = label_tune[shuffle_array]
ground_xy_tune = ground_xy_tune[shuffle_array]

# 转张量
label_tune = torch.from_numpy(label_tune).type(torch.LongTensor)
label_tlt_1 = torch.from_numpy(label_tlt_1).type(torch.LongTensor)
label_tlt_2 = torch.from_numpy(label_tlt_2).type(torch.LongTensor)
ground_xy_tune = torch.from_numpy(ground_xy_tune).type(torch.LongTensor)
ground_xy_tlt_1 = torch.from_numpy(ground_xy_tlt_1).type(torch.LongTensor)
ground_xy_tlt_2 = torch.from_numpy(ground_xy_tlt_2).type(torch.LongTensor)
ground_xy_allData = torch.from_numpy(ground_xy_allData).type(torch.LongTensor)

print('训练样本数：', len(label_tune))
print('验证样本数：',len(label_tlt_1))
print('测试样本数：', len(label_tlt_2))

# 数据归一化
ms4 = to_tensor(ms4_np)
pan = to_tensor(pan_np)
pan = np.expand_dims(pan, axis=0)  # 二维数据进网络前要加一维
ms4 = np.array(ms4).transpose((2, 0, 1))  # 调整通道 chw

# 转换类型
ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
pan = torch.from_numpy(pan).type(torch.FloatTensor)

# 数据集对象被抽象为Dataset类
# 训练集、测试集对象
class MyData(Dataset):
    def __init__(self, MS4, Pan, Label, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan
        self.train_labels = Label
        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]  # 取出当前标签的坐标
        x_pan = int(4 * x_ms)
        y_pan = int(4 * y_ms)
        # 切出中心点的周围部分区域
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]  # dim：chw

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]  # 坐标信息

        target = self.train_labels[index]  # 当前像素点的类别信息

        return image_ms, image_pan, target, locate_xy

    def __len__(self):
        return len(self.gt_xy)

# 测试集对象
class MyData1(Dataset):
    def __init__(self, MS4, Pan, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan

        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)  # 计算不可以在切片过程中进行
        y_pan = int(4 * y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        return image_ms, image_pan, locate_xy

    def __len__(self):
        return len(self.gt_xy)

train_data = MyData(ms4, pan, label_tune, ground_xy_tune, Ms4_patch_size)
test_data_1 = MyData(ms4, pan, label_tlt_1, ground_xy_tlt_1, Ms4_patch_size)
test_data_2 = MyData(ms4, pan, label_tlt_2, ground_xy_tlt_2, Ms4_patch_size)

all_data = MyData1(ms4, pan, ground_xy_allData, Ms4_patch_size)

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader_1 = DataLoader(dataset=test_data_1, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader_2 = DataLoader(dataset=test_data_2, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
all_data_loader = DataLoader(dataset=all_data, batch_size=BATCH_SIZE, shuffle=False,num_workers=0)


# 初始化迁移学习网络模型
model = Net(num_classes=Categories_Number_train).to(device)
# 初始化优化器
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#定义训练方法
def train_model(model, train_loader, optimizer, epoch):
    model.train()
    correct = 0.0
    total=0
    train_bar=tqdm(train_loader,desc="train",colour='yellow')
    for step, (ms, pan, label, _) in enumerate(train_bar):
        ms, pan, label = ms.to(device), pan.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(ms, pan)
        total+=output.size(0)
        pred_train = output.max(1, keepdim=True)[1]  # 抽出最大值对应的下标
        correct += pred_train.eq(label.view_as(pred_train).long()).sum().item()  # 求 TP
        loss = F.cross_entropy(output, label.long())  # 交叉熵作为loss
            #定义反向传播
        loss.backward()
            #定义优化
        optimizer.step()
        train_bar.set_description(f"Epoch[{epoch}/{EPOCH}]")
        train_bar.set_postfix(train_loss=loss.item(),train_acc=correct * 100.0 /total)
        # if step % 1 == 0:
        #     print("Train Epoch: {} \t Loss : {:.6f} \t step: {} ".format(epoch, loss.item(), step))
    # print("Train Accuracy: {:.6f}".format(correct * 100.0 / len(train_loader.dataset)))

#定义测试方法
def test_model(model, test_loader):
    model.eval()
    correct = 0.0
    test_loss = 0.0
    with torch.no_grad():
        for data, data1, target, _ in test_loader:
            data, data1, target = data.to(device), data1.to(device), target.to(device)
            output = model(data, data1)
            test_loss += F.cross_entropy(output, target.long()).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred).long()).sum().item()
        test_loss = test_loss / len(test_loader.dataset)
        print("test-average loss: {:.4f}, Accuracy:{:.3f} \n".format(
            test_loss, 100.0 * correct / len(test_loader.dataset)
        ))


def test_second(model, test_loader, mode='2'):
    model.eval()
    test_loss = 0
    correct = 0.0
    test_matrix = np.zeros([Categories_Number_train, Categories_Number_train])
    loop = tqdm(test_loader, leave=True)
    with torch.no_grad():
        for batch_idx, (data1, data2, target, _) in enumerate(loop):
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            output = model(data1, data2)
            # test_loss += nn.CrossEntropyLoss(output, target.long())
            test_loss += F.cross_entropy(output, target.long()).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred).long()).sum().item()
            for i in range(len(target)):
                test_matrix[int(pred[i].item())][int(target[i].item())] += 1
            loop.set_postfix(mode='test')
        loop.close()
        test_loss = test_loss / len(test_loader.dataset)
        print("test-average loss: {:.4f}, Accuracy:{:.3f} \n".format(
            test_loss, 100.0 * correct / len(test_loader.dataset)
        ))
    return test_matrix

def test_third(model, test_loader):
    model.eval()
    correct = 0.0
    total_1=0
    test_bar=tqdm(test_loader,desc="test",colour='green')
    test_metric=np.zeros([Categories_Number_train,Categories_Number_train])
    with torch.no_grad():
        for data, data1, target, _  in test_bar:
            data, data1, target= data.to(device), data1.to(device), target.to(device)
            output= model(data,data1)
            total_1+=output.size(0)
            test_loss = F.cross_entropy(output, target.long()).item()
            pred = output.max(1, keepdim=True)[1]
            for i in range(len(target)):
                test_metric[int(pred[i].item())][int(target[i].item())]+=1
            correct += pred.eq(target.view_as(pred).long()).sum().item()
            test_bar.set_description(f"Epoch[{epoch}/{EPOCH}]")
            test_bar.set_postfix(test_loss=test_loss,test_acc=100.0 * correct / total_1)
        # print("test Accuracy:{:.3f} \n".format( 100.0 * correct / len(test_loader.dataset)))
    b=np.sum(test_metric,axis=0)
    accuracy=[]
    c=0
    for i in range(0,Categories_Number_train):
        a=test_metric[i][i]/b[i]
        accuracy.append(a)
        c+=test_metric[i][i]
        print('category {0:d}: {1:f}'.format(i,a))
    average_accuracy = np.mean(accuracy)
    overall_accuracy = c/np.sum(b,axis=0)
    kappa_coefficient = kappa(test_metric)
    print('AA: {0:f}'.format(average_accuracy))
    print('OA: {0:f}'.format(overall_accuracy))
    print('KAPPA: {0:f}'.format(kappa_coefficient))
    return 100.0 * correct / len(test_loader.dataset)


def aa_oa(matrix):
    accuracy = []
    b = np.sum(matrix, axis=0)
    c = 0
    on_display = []
    for i in range(1, matrix.shape[0]):
        a = matrix[i][i]/b[i]
        c += matrix[i][i]
        accuracy.append(a)
        on_display.append([b[i], matrix[i][i], a])
        print("Category:{}. Overall:{}. Correct:{}. Accuracy:{:.6f}".format(i, b[i], matrix[i][i], a))
    aa = np.mean(accuracy)
    oa = c / np.sum(b, axis=0)
    k = kappa(matrix)
    print("OA:{:.6f} AA:{:.6f} Kappa:{:.6f}".format(oa, aa, k))
    return [aa, oa, k, on_display]

def kappa(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    # print(po, pe)
    return (po - pe) / (1 - pe)

# 训练 & 测试
for epoch in range(1, EPOCH+1):
    train_model(model,  train_loader, optimizer, epoch)
    if epoch==EPOCH:
        test_matrix =test_third(model,test_loader_2)
        print(test_matrix)
        torch.save(model,'./sh/tune/sh_18_35.pkl')

# 加载模型
cnn = torch.load('./南京/南京tune_model/model_10_10.pkl')
cnn.to(device)

# 上色
class_count = np.zeros(7)
out_clour = np.zeros((800, 830, 3))
def clour_model(cnn, all_data_loader):
    for step, (ms4, pan, gt_xy) in enumerate(all_data_loader):
        ms4 = ms4.to(device)
        pan = pan.to(device)
        with torch.no_grad():
            output = cnn(ms4, pan)
        pred_y = torch.max(output, 1)[1].to(device).data.squeeze()
        pred_y_numpy = pred_y.cpu().numpy()
        #print("pred_y###",pred_y_numpy)
        gt_xy = gt_xy.numpy()
        for k in range(len(gt_xy)):
            if pred_y_numpy[k] == 0:
                class_count[0] = class_count[0] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [203, 192, 255]
            elif pred_y_numpy[k] == 1:
                class_count[1] = class_count[1] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [14, 132, 241]
            elif pred_y_numpy[k] == 2:
                class_count[2] = class_count[2] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [255, 255, 0]
            elif pred_y_numpy[k] == 3:
                class_count[3] = class_count[3] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [0, 0, 255]
            elif pred_y_numpy[k] == 4:
                class_count[4] = class_count[4] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [51, 102, 103]
            elif pred_y_numpy[k] == 5:
                class_count[5] = class_count[5] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [0, 255, 0]
            elif pred_y_numpy[k] == 6:
                class_count[6] = class_count[6] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [255, 0, 0]
    print(class_count)
    cv2.imwrite("back_bone1.png", out_clour)

# 上色
clour_model(model,  all_data_loader)
# 打印时间
import datetime

current_time = datetime.datetime.now()
print("current_time:    " + str(current_time))
