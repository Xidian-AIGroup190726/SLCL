import torch
def d_pos(e_distance):
    # 获取对角线元素
    # d_pos= torch.unsqueeze(torch.diagonal(e_distance), dim=1)#需要一维则dim=0
    d_pos=torch.diag(e_distance)
    return d_pos

def d_neg(e_distance):
    n = min(e_distance.size(0), e_distance.size(1))
    d_neg=e_distance.clone()
    # 将对角线元素设置为零
    d_neg[range(n), range(n)] = 0
    # d_neg=e_distance
    return d_neg

#计算负样本权重矩阵
def weight(neg,q):
    max_value = neg.max()
    # diag=torch.diag(neg)
    # diag_arg=diag.mean()
    # neg_max, _ = torch.topk(neg.flatten(), min(3, len(neg.flatten())), largest=True)
    # max_value=neg_max.mean()
    neg.fill_diagonal_(2)
    min_value = neg.min()
    # neg_min, _ = torch.topk(neg.flatten(), min(3, len(neg.flatten())), largest=False)
    # min_value=neg_min.mean()
    neg.fill_diagonal_(0)
    # 计算阈值
    threshold = (max_value + min_value) / 4
    # print(threshold)
    # 小于阈值的元素置为0，大于等于阈值的元素置为1,得到权重矩阵
    neg_weight = torch.where(neg < threshold, q, 1)
    neg_weight.fill_diagonal_(0)
    return neg_weight

#计算D_neg
def D_neg(neg,weight):
    D_neg=neg*weight
    return D_neg

#计算D
def D(D_neg):
    # D = torch.tensor([row[row.nonzero(as_tuple=True)].min() for row in D_neg])

    # result = []
    result_1=[]
    for row in D_neg:
        nonzero_vals = row[row != 0]  # 提取非零元素
        # if nonzero_vals.numel() == 0:  # 如果没有非零元素
        #     continue  # 跳过该行
        # topk_vals, _ = torch.topk(nonzero_vals, min(K, nonzero_vals.numel()), largest=False)  # 找到最小的k个非零元素
        # result.extend(topk_vals.tolist())# 将行的结果添加到列表中
        result_1.extend(nonzero_vals.tolist())
    return torch.tensor(result_1)

# n=torch.tensor([[1,2,4,6],
#                 [1,4,0,3],
#                 [4,5,8,9]])
# m,b=D(n,3)
# print(m.flatten(),b)
