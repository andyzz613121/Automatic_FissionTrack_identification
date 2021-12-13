import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
def loss_with_hed(input, edge, label, threshold=0.5):
    n, c, h, w = input.size()
    log_cross = input.transpose(0, 1).contiguous().view(1, c, -1)
    edge = edge.view(1, -1).float()
    label = label.view(1, -1).long()

    weights_EDGE = edge.cpu().detach().numpy()
    weights_EDGE = torch.from_numpy(weights_EDGE).cuda()
    weights_EDGE = weights_EDGE/torch.max(weights_EDGE)

    weights_class = np.zeros((label.shape[0], label.shape[1])).astype(np.float32)
    weights_class = torch.from_numpy(weights_class).cuda()
    red_index = (label == 2) #low angle tracks
    pos_index = (label == 1) #high angle tracks
    neg_index = (label == 0) #background
    red_num = red_index.sum().float()
    pos_num = pos_index.sum().float()
    neg_num = neg_index.sum().float()
    sum_num = pos_num + neg_num + red_num

    pos_rate = 1-(pos_num/sum_num)
    neg_rate = 1-(neg_num/sum_num)
    red_rate = 1-(red_num/sum_num)
   
    weights_class[pos_index] = pos_rate
    weights_class[neg_index] = neg_rate
    weights_class[red_index] = red_rate
    
    weights_class = weights_class/torch.max(weights_class)

    weight = torch.add(weights_class, weights_EDGE)
    m = nn.LogSoftmax(dim=1)
    labeln_1 = torch.unsqueeze(label[0], 1) 
    one_hot = torch.zeros(label.shape[1], 3).cuda().scatter_(1, labeln_1, 1)
    one_hot = one_hot.transpose(0, 1)
    log_cross = m(log_cross)

    loss_ce = torch.mul(one_hot, log_cross[0])
    
    loss_ce = torch.mul(loss_ce, weight[0])
    loss_ce = -1 * torch.sum(loss_ce)/label.shape[1]
    return loss_ce
