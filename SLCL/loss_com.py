import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#定义比较器
def zero_con(a):
    zero=torch.tensor(0,dtype=torch.float)
    if a>zero:
        return a
    else:
        return zero

#cross loss
def loss_cross(d_pos,D_1,D_2,D_3,K,a):
    loss_cross_1, _ = torch.topk(d_pos.flatten(), K, largest=False)
    loss_cross_1=loss_cross_1.mean()
    D_1_min, _ = torch.topk(D_1.flatten(), min(K,len(D_1.flatten())), largest=False)
    # print(D_1_min)
    D_1_min=D_1_min.mean()
    D_1_max, _ = torch.topk(D_1.flatten(), min(K,len(D_1.flatten())), largest=True)
    D_1_max = D_1_max.mean()
    D_2_min, _ = torch.topk(D_2.flatten(), min(K,len(D_2.flatten())), largest=False)
    D_2_min = D_2_min.mean()
    D_2_max, _ = torch.topk(D_2.flatten(), min(K,len(D_2.flatten())), largest=True)
    D_2_max = D_2_max.mean()
    D_3_min, _ = torch.topk(D_3.flatten(), min(K,len(D_3.flatten())), largest=False)
    D_3_min = D_3_min.mean()
    D_3_max, _ = torch.topk(D_3.flatten(), min(K,len(D_3.flatten())), largest=True)
    D_3_max = D_3_max.mean()
    loss_cross_2_sum=torch.relu(D_2_max-D_1_max)+torch.relu(D_2_min-D_1_min)
    # print(loss_cross_2_sum)
    loss_cross_2=loss_cross_2_sum/2#注意一下是否真的大于0
    loss_cross_3_sum=torch.relu(D_2_max-D_3_max)+torch.relu(D_2_min-D_3_min)
    # print(loss_cross_3_sum)
    loss_cross_3 = loss_cross_3_sum/2
    # loss_cross=a*loss_cross_1+(1-a)*(loss_cross_2/2+loss_cross_3/2)
    loss_cross=(loss_cross_2+loss_cross_3)/2
    return loss_cross

def cross(D_1,D_2,D_3):
    mse_1=torch.mean((D_1/0.1-D_2/0.1)**2)
    mse_2=torch.mean((D_2/0.1-D_3/0.1)**2)
    mse_3=torch.mean((D_3/0.1-D_1/0.1)**2)
    return (mse_1+mse_2+mse_3)/3


#margin loss
def margin(d_pos,D_1,D_2,D_3,K):
    d_pos_max, _ = torch.topk(d_pos.flatten(), K, largest=True)
    # print(d_pos_max)
    d_pos_max=d_pos_max.mean()
    # print(d_pos_max)
    D_1_min, _ = torch.topk(D_1.flatten(), K, largest=False)
    # print(D_1_min)
    D_1_min=D_1_min.mean()
    # print(D_1_min)
    D_2_min, _ = torch.topk(D_2.flatten(), K, largest=False)
    # print(D_2_min)
    D_2_min = D_2_min.mean()
    # print(D_2_min)
    D_3_min, _ = torch.topk(D_3.flatten(), K, largest=False)
    # print(D_3_min)
    D_3_min = D_3_min.mean()
    # print(D_3_min)
    margin_1=torch.relu(d_pos_max-D_1_min)
    # print(margin_1)
    margin_2 = torch.relu(d_pos_max - D_2_min)
    # print(margin_2)
    margin_3 = torch.relu(d_pos_max - D_3_min)
    # print(margin_3)
    margin=(margin_1+margin_2+margin_3)/3
    # print(margin)
    return margin


#contrastive loss
def loss_contrastive(z1,z2,weight1,weight2,weight3,temperature):
    dot_1=torch.mm(z1,z2.t())
    dot_2=torch.mm(z2,z1.t())
    dot_3=torch.mm(z1,z1.t())
    dot_4=torch.mm(z2,z2.t())
    positive_loss_1=torch.exp(dot_1/temperature)
    positive_loss_2=torch.exp(dot_2/temperature)
    negative_loss_1=torch.exp(dot_3/temperature)
    negative_loss_2=torch.exp(dot_4/temperature)
    # weight2_2=1-weight2
    # weight1_1=1-weight1
    # weight3_3=1-weight3
    positive_loss_1_weight=positive_loss_1*weight2
    # positive_loss_1_weight_d=positive_loss_1*weight2_2
    positive_loss_2_weight=positive_loss_2*weight2.t()
    # positive_loss_2_weight_d=positive_loss_2*weight2_2.t()
    negative_loss_1_weight = negative_loss_1 * weight1
    # negative_loss_1_weight_d=negative_loss_1*weight1_1
    negative_loss_2_weight = negative_loss_2 * weight3
    # negative_loss_2_weight_d=negative_loss_2*weight3_3
    loss_1=-torch.log(positive_loss_1.diag()/(positive_loss_1_weight.sum(1)+negative_loss_1_weight.sum(1)))
    loss_2=-torch.log(positive_loss_2.diag()/(positive_loss_2_weight.sum(1)+negative_loss_2_weight.sum(1)))
    # loss_1 = (positive_loss_1_weight_d + negative_loss_1_weight_d) / (
    #             positive_loss_1_weight.sum(1) + negative_loss_1_weight.sum(1))
    # loss_2 = (positive_loss_2_weight_d + negative_loss_2_weight_d) / (
    #             positive_loss_2_weight.sum(1) + negative_loss_2_weight.sum(1))
    # loss_1_indices = torch.nonzero(loss_1)
    # loss_1_values = loss_1[loss_1_indices[:, 0], loss_1_indices[:, 1]]
    # loss_2_indices = torch.nonzero(loss_2)
    # loss_2_values = loss_2[loss_2_indices[:, 0], loss_2_indices[:, 1]]
    # loss_1_log=-torch.log(loss_1_values)
    # loss_2_log=-torch.log(loss_2_values)
    # loss_contrastive_1=loss_1_log.mean()
    loss_contrastive_1=loss_1.mean()
    # loss_contrastive_2=loss_2_log.mean()
    loss_contrastive_2=loss_2.mean()
    loss_contrastive=(loss_contrastive_1+loss_contrastive_2)/2
    return loss_contrastive

def l_contrastive_classic(z1,z2,temperature):
    dot_1 = torch.mm(z1, z2.t())
    dot_2 = torch.mm(z2, z1.t())
    # dot_3 = torch.mm(z1, z1.t())
    # dot_4 = torch.mm(z2, z2.t())
    positive_loss_1 = torch.exp(dot_1 / temperature)
    positive_loss_2 = torch.exp(dot_2 / temperature)
    loss_1 = -torch.log(positive_loss_1.diag() / (positive_loss_1.sum(1)))
    loss_2 = -torch.log(positive_loss_2.diag() / (positive_loss_2.sum(1)))
    loss_contrastive_1 = loss_1.mean()
    loss_contrastive_2 = loss_2.mean()
    loss_contrastive = (loss_contrastive_1 + loss_contrastive_2) / 2
    return loss_contrastive

def loss_contrastive_group(z1,z2,temperature):
    dot_1=torch.mm(z1,z2.t())
    dot_2=torch.mm(z2,z1.t())
    dot_3=torch.mm(z1,z1.t())
    dot_4=torch.mm(z2,z2.t())
    positive_loss_1=torch.exp(dot_1/temperature)
    positive_loss_2=torch.exp(dot_2/temperature)
    negative_loss_1=torch.exp(dot_3/temperature)
    negative_loss_2=torch.exp(dot_4/temperature)
    positive_loss_1_weight=positive_loss_1
    positive_loss_2_weight=positive_loss_2
    negative_loss_1_weight = negative_loss_1
    negative_loss_2_weight = negative_loss_2
    loss_1=-torch.log(positive_loss_1.diag()/(positive_loss_1_weight.sum(1)+negative_loss_1_weight.sum(1)))
    loss_2=-torch.log(positive_loss_2.diag()/(positive_loss_2_weight.sum(1)+negative_loss_2_weight.sum(1)))
    loss_contrastive_1=loss_1.mean()
    loss_contrastive_2=loss_2.mean()
    loss_contrastive=(loss_contrastive_1+loss_contrastive_2)/2
    return loss_contrastive

def loss_contrastive_weight(z1,z2,weight2,temperature):
    dot_1=torch.mm(z1,z2.t())
    dot_2=torch.mm(z2,z1.t())
    # dot_3=torch.mm(z1,z1.t())
    # dot_4=torch.mm(z2,z2.t())
    positive_loss_1=torch.exp(dot_1/temperature)
    positive_loss_2=torch.exp(dot_2/temperature)
    # negative_loss_1=torch.exp(dot_3/temperature)
    # negative_loss_2=torch.exp(dot_4/temperature)
    positive_loss_1_weight=positive_loss_1*weight2
    positive_loss_2_weight=positive_loss_2*weight2.t()
    loss_1=-torch.log(positive_loss_1.diag()/(positive_loss_1_weight.sum(1)))
    loss_2=-torch.log(positive_loss_2.diag()/(positive_loss_2_weight.sum(1)))
    loss_contrastive_1=loss_1.mean()
    loss_contrastive_2=loss_2.mean()
    loss_contrastive=(loss_contrastive_1+loss_contrastive_2)/2
    return loss_contrastive

#hard sample loss
def fpn_loss(d_pos,D):
    min=torch.min(D)
    max=torch.max(d_pos)
    # print(min)
    # print(min.type())
    # print(max.type())
    d_pos_judge=d_pos-min
    D_judge=max-D
    # print(d_pos_judge.type())
    # print(D_judge.type())
    # 获取大于0的元素
    fp_0 = torch.masked_select(d_pos_judge, d_pos_judge > 0)
    fn_0=torch.masked_select(D_judge, D_judge > 0)
    # 对大于0的元素进行按降序排序
    # print(fp_0.type())
    # print(fn_0.type())
    r1, rank_1 = torch.sort(fp_0, descending=True)
    r2, rank_2 = torch.sort(fn_0, descending=True)
    # print(rank_1.type())
    # print(rank_2.type())
    # 创建一个新的张量来存储排序后的元素的编号
    ranked_1 = torch.zeros_like(rank_1,dtype=torch.float)
    ranked_2 = torch.zeros_like(rank_2,dtype=torch.float)
    # print(ranked_1.type())
    # print(ranked_2.type())
    for i in range(len(rank_1)):
        ranked_1[rank_1[i]] = i + 1
    for i in range(len(rank_2)):
        ranked_2[rank_2[i]] = i + 1
    weight_fp=1/torch.exp(ranked_1)
    weight_fn=1/torch.exp(ranked_2)
    # print(weight_fp.type())
    # print(weight_fn.type())
    fp_sum=weight_fp*(fp_0/0.1)
    fp_sum=fp_sum.sum()
    # print(fp_sum.type())
    fn_sum=weight_fn*(fn_0/0.1)
    fn_sum=fn_sum.sum()
    # print(fn_sum.type())
    fpn_loss=torch.log(torch.relu(fp_sum+fn_sum)+1e-8)
    # print(fpn_loss.type())
    # fpn_loss=torch.log(fp_sum+fn_sum)
    return fpn_loss

def loss_hard_sample(d_pos,D_1,D_2,D_3,q,p):
    loss_hard_sample_1=fpn(d_pos,D_1.flatten(),q,p)
    loss_hard_sample_2 = fpn(d_pos, D_2.flatten(),q,p)
    loss_hard_sample_3 = fpn(d_pos, D_3.flatten(),q,p)
    loss_hard_sample=(loss_hard_sample_1+loss_hard_sample_2+loss_hard_sample_3)/3
    return loss_hard_sample



def fpn(d_pos,D,q,p):
    if D.numel()==0 or d_pos.numel()==0:
        fpn_loss=torch.tensor(0)
    else:
        min = torch.min(D)
        max = torch.max(d_pos)
        d_pos_judge = d_pos - min
        D_judge = max - D
        # print(d_pos_judge.type())
        # print(D_judge.type())
        # 获取大于0的元素
        fp_0 = torch.masked_select(d_pos_judge, d_pos_judge > 0)
        fn_0 = torch.masked_select(D_judge, D_judge > 0)
        # 对大于0的元素进行按降序排序
        # print(fp_0.type())
        # print(fn_0.type())
        r1, _ = torch.sort(fp_0, descending=True)
        r2, _ = torch.sort(fn_0, descending=False)
        r3, _ = torch.sort(fn_0, descending=True)

        r1 = r1.to(device)
        # print(r1)
        r2 = r2.to(device)
        r3 = r3.to(device)

        lenth_1 = int(len(r2) * q)
        lenth_2 = int(len(r3) * q * p)

        fn_small = r2[:lenth_1]
        fn_large = r3[:lenth_2]

        weight_1 = torch.arange(1, len(r1) + 1)
        weight_1 = weight_1.to(device)
        weight_2 = torch.arange(1, len(fn_small) + 1)
        weight_2 = weight_2.to(device)
        weight_3 = torch.arange(len(fn_large) + len(fn_small), len(fn_small), -1)
        weight_3 = weight_3.to(device)

        weight_fp = 1 / torch.exp(weight_1)
        weight_fn = 1 / torch.exp(weight_2)
        weight_fn_1 = 1 / torch.exp(weight_3)

        fp_sum = weight_fp * (r1 / 0.1)
        fp_sum = fp_sum.sum()
        fn_sum = weight_fn * (fn_small / 0.1)
        fn_sum = fn_sum.sum()
        fn_sum_1 = weight_fn_1 * (fn_large / 0.1)
        fn_sum_1 = fn_sum_1.sum()

        # loss_1=torch.log(fp_sum+1)
        # loss_2=torch.log(fn_sum+1)
        # loss_3=torch.log(fn_sum_1+1)

        # fpn_loss=loss_1.mean()+loss_2.mean()+loss_3.mean()

        fpn_loss = torch.log(fp_sum + fn_sum + fn_sum_1 + 1)

    return fpn_loss

