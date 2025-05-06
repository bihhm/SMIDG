"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader

import tqdm


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, te, features, is_unsup = None,epoch = None,labels=None,fusion=False ,mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        self.temperature=te
        self.base_temperature=te
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        features = F.normalize(features, p=2, dim=1)
        features = features.unsqueeze(1)
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        #这个是开启有监督和自监督融合的epoch
        mid_batch = int(batch_size/2)
        if epoch<=10 and fusion:
            
            temp_diag = torch.eye(mid_batch).to(device)
            mask[:mid_batch,mid_batch:] = temp_diag
            mask[mid_batch:,:mid_batch] = temp_diag
        if is_unsup:  #开启自监督，反之有监督
            temp_diag = torch.eye(mid_batch).to(device)
            mask[:mid_batch,mid_batch:] = temp_diag
            mask[mid_batch:,:mid_batch] = temp_diag
        # print(mask)

        # contrast count is the number of augmented views
        contrast_count = features.shape[1]
        # contrast feature: the concatenation of all views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        neg_mask = anchor_dot_contrast > (0.4 / self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # print(logits_mask)
        mask = mask * logits_mask
        # print(mask.sum(1))

        # print(anchor_feature.shape)

        # compute log_prob
        if is_unsup:
            exp_logits = torch.exp(logits) * logits_mask * neg_mask # this is the numerator -> positive
        else:
            exp_logits = torch.exp(logits) * logits_mask # this is the numerator -> positive
        # print(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) # this is the denom -> negative
        # print(log_prob)

        # compute mean of log-likelihood over positive
        # Prevent the nan loss by adding a small amount of number in the denom
        eps = 1e-12
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + eps)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


# AutomaticMetricLoss
class AutomaticMetricLoss(nn.Module):
    def __init__(self, num=2, init_weight=1.0, min_weights=[0,0]):
        super(AutomaticMetricLoss, self).__init__()
        self.num = num
        self.min_weights = min_weights
        params = torch.ones(num, requires_grad=True) * init_weight
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        weights = []
        bias = []
        for i, loss in enumerate(x):
            weights_i = 0.5 / (self.params[i] ** 2) + self.min_weights[i]
            bias_i = torch.log(1 + self.params[i] ** 2)
            loss_sum += weights_i * loss + bias_i
            weights.append(weights_i)
            bias.append(bias_i)
        return loss_sum, weights, bias





def collect_feature_domain(data_loader_train: DataLoader,data_loader_test: DataLoader, feature_extractor: nn.Module,
                    device: torch.device, max_num_features=None) -> torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `feature_extractor` to collect features
    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device)
        max_num_features (int): The max number of features to return
    Returns:
        Features in shape (min(len(data_loader), max_num_features * mini-batch size), :math:`|\mathcal{F}|`).
    """
    feature_extractor.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for i, (batch,label,domain) in enumerate(data_loader_train):
            if max_num_features is not None and i >= max_num_features:
                break
            batch = torch.cat(batch, dim=0).to(device)
            domains = torch.cat(domain, dim=0).to(device)
            # max_value = torch.max(domains)
            # print(max_value)
            # print(domains.shape)
            feature,_ = feature_extractor(batch)
            all_features.append(feature)
            all_labels.append(domains)
        # 假设测试集的域标签是1
        test_domain_label = 1
        for it, (batchs, domain) in enumerate(data_loader_test):
            
            batch, labels, domains = batchs[0].to(device), batchs[1].to(device), domain.to(device)
            
            # print(batch.shape)
            # print(labels.shape)
            # print(labels)
            # print(domain.shape)
            # print(domain)
            # time.sleep(20)
            domains = domains + 3
            # if self.args.target in pacs_dataset:
            # domains = torch.cat(domain, dim=0).to(device)
            # domains = torch.full((batch.size(0),), test_domain_label, dtype=torch.long, device=device)
            feature,_ = feature_extractor(batch)
            all_features.append(feature)
            all_labels.append(domains)
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)

def collect_feature(data_loader_train: DataLoader, feature_extractor: nn.Module,
                    device: torch.device, max_num_features=None) -> torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `feature_extractor` to collect features
    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device)
        max_num_features (int): The max number of features to return
    Returns:
        Features in shape (min(len(data_loader), max_num_features * mini-batch size), :math:`|\mathcal{F}|`).
    """
    feature_extractor.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for i, (batch,label,domain) in enumerate(data_loader_train):
            if max_num_features is not None and i >= max_num_features:
                break
            batch = torch.cat(batch, dim=0).to(device)
            labels = torch.cat(label, dim=0).to(device)
            # print(feature_extractor)
            # print(batch.shape)
            feature,_ = feature_extractor(batch)
            # print(feature.shape)
            all_features.append(feature)
            all_labels.append(labels)
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# class SupConLoss(nn.Module):

#     def __init__(self, temperature=0.5, scale_by_temperature=True):
#         super(SupConLoss, self).__init__()
#         self.temperature = temperature
#         self.scale_by_temperature = scale_by_temperature

#     def forward(self, features, labels=None, mask=None):
#         """
#         输入:
#             features: 输入样本的特征，尺寸为 [batch_size, hidden_dim].
#             labels: 每个样本的ground truth标签，尺寸是[batch_size].
#             mask: 用于对比学习的mask，尺寸为 [batch_size, batch_size], 如果样本i和j属于同一个label，那么mask_{i,j}=1 
#         输出:
#             loss值
#         """
#         device = (torch.device('cuda')
#                   if features.is_cuda
#                   else torch.device('cpu'))
#         features = F.normalize(features, p=2, dim=1)
#         batch_size = features.shape[0]
#         # 关于labels参数
#         if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
#             raise ValueError('Cannot define both `labels` and `mask`') 
#         elif labels is None and mask is None: # 如果没有labels，也没有mask，就是无监督学习，mask是对角线为1的矩阵，表示(i,i)属于同一类
#             mask = torch.eye(batch_size, dtype=torch.float32).to(device)
#         elif labels is not None: # 如果给出了labels, mask根据label得到，两个样本i,j的label相等时，mask_{i,j}=1
#             labels = labels.contiguous().view(-1, 1)
#             if labels.shape[0] != batch_size:
#                 raise ValueError('Num of labels does not match num of features')
#             mask = torch.eq(labels, labels.T).float().to(device)
#         else:
#             mask = mask.float().to(device)
#         '''
#         示例: 
#         labels: 
#             tensor([[1.],
#                     [2.],
#                     [1.],
#                     [1.]])
#         mask:  # 两个样本i,j的label相等时，mask_{i,j}=1
#             tensor([[1., 0., 1., 1.],
#                     [0., 1., 0., 0.],
#                     [1., 0., 1., 1.],
#                     [1., 0., 1., 1.]]) 
#         '''
#         # compute logits
#         anchor_dot_contrast = torch.div(
#             torch.matmul(features, features.T),
#             self.temperature)  # 计算两两样本间点乘相似度
#         # for numerical stability
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#         logits = anchor_dot_contrast - logits_max.detach()
#         exp_logits = torch.exp(logits)
#         '''
#         logits是anchor_dot_contrast减去每一行的最大值得到的最终相似度
#         示例: logits: torch.size([4,4])
#         logits:
#             tensor([[ 0.0000, -0.0471, -0.3352, -0.2156],
#                     [-1.2576,  0.0000, -0.3367, -0.0725],
#                     [-1.3500, -0.1409, -0.1420,  0.0000],
#                     [-1.4312, -0.0776, -0.2009,  0.0000]])       
#         '''
#         # 构建mask 
#         logits_mask = torch.ones_like(mask).to(device) - torch.eye(batch_size).to(device)     
#         temp_diag = torch.eye(int(batch_size/2)).to(device)
#         mask[:32,32:] = temp_diag
#         mask[32:,:32] = temp_diag
#         positives_mask = mask * logits_mask
#         negatives_mask = 1. - mask
#         '''
#         但是对于计算Loss而言，(i,i)位置表示样本本身的相似度，对Loss是没用的，所以要mask掉
#         # 第ind行第ind位置填充为0
#         得到logits_mask:
#             tensor([[0., 1., 1., 1.],
#                     [1., 0., 1., 1.],
#                     [1., 1., 0., 1.],
#                     [1., 1., 1., 0.]])
#         positives_mask:
#         tensor([[0., 0., 1., 1.],
#                 [0., 0., 0., 0.],
#                 [1., 0., 0., 1.],
#                 [1., 0., 1., 0.]])
#         negatives_mask:
#         tensor([[0., 1., 0., 0.],
#                 [1., 0., 1., 1.],
#                 [0., 1., 0., 0.],
#                 [0., 1., 0., 0.]])
#         '''        
#         num_positives_per_row  = torch.sum(positives_mask , axis=1) # 除了自己之外，正样本的个数  [2 0 2 2]       
#         denominator = torch.sum(
#         exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
#             exp_logits * positives_mask, axis=1, keepdims=True)  
        
#         log_probs = logits - torch.log(denominator)
#         if torch.any(torch.isnan(log_probs)):
#             raise ValueError("Log_prob has nan!")
        

#         log_probs = torch.sum(
#             log_probs*positives_mask , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
#         '''
#         计算正样本平均的log-likelihood
#         考虑到一个类别可能只有一个样本，就没有正样本了 比如我们labels的第二个类别 labels[1,2,1,1]
#         所以这里只计算正样本个数>0的    
#         '''
#         # loss
#         loss = -log_probs
#         if self.scale_by_temperature:
#             loss *= self.temperature
#         loss = loss.mean()
#         return loss