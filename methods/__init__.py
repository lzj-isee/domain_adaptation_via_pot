import torch.nn.functional as F
import torch
import numpy as np
from collections import Counter


@torch.no_grad()
def update_prototype(prototype, feature_s, label_s_onehot, feature_t, index_unmatch):
    num_unmatch = len(index_unmatch)
    count = torch.cat([label_s_onehot.sum(0), torch.ones([1], device = 'cuda') * num_unmatch])
    feature_ext = torch.cat([feature_s, feature_t[index_unmatch]], dim = 0)
    label_ext = torch.cat([label_s_onehot, torch.zeros((num_unmatch, label_s_onehot.size(1)), device = 'cuda')], dim = 0)
    label_ext = torch.cat([label_ext, torch.zeros((label_ext.size(0), 1), device = 'cuda')], dim = 1)
    label_ext[label_s_onehot.size(0):, -1] = 1
    prototype = 0.9 * prototype + 0.1 * torch.matmul(label_ext.t(), feature_ext) / \
        torch.where(count > 0, count, torch.ones(1, device = 'cuda')).view(-1, 1)
    return prototype

@torch.no_grad()
def find_match(gamma):
    gamma = gamma.cpu().numpy()
    threshold = gamma.max() / 2
    loc_x, loc_y = np.where(gamma > threshold)
    return loc_x, loc_y

@torch.no_grad()
def match_information(gamma, label_s, label_t):
    loc_x, loc_y = find_match(gamma)
    match_count = 0
    for x, y, in zip(loc_x, loc_y):
        if label_s[x] == label_t[y]: match_count += 1
    freq_s = Counter(np.sort(label_s))
    freq_t = Counter(np.sort(label_t))
    total_count = 0
    for label in freq_s:
        if label in freq_t:
            total_count += min(freq_s[label], freq_t[label])
        else:
            pass
    return match_count / len(loc_x) * 100, total_count / len(label_s) * 100, match_count / total_count * 100

@torch.no_grad()
def threshold_calc(gamma, label_one_hot, pred_s, pred_t):
    pred_s = pred_s[label_one_hot == 1]
    sigmoids = torch.cat([pred_s, 2 - pred_s])
    threshold = torch.clamp(1 - 3 * sigmoids.std(), 0, 1)
    return threshold

@torch.no_grad()
def threshold_calc2(gamma, label_one_hot, pred_s, pred_t):
    threshold_gamma = gamma.max() / 2
    sigmoids = torch.where(gamma > threshold_gamma, 1, 0) * torch.matmul(label_one_hot, pred_t.detach().t())
    sigmoids = sigmoids.masked_select(sigmoids > 0)
    sigmoids = torch.cat([sigmoids, 2 - sigmoids])
    threshold = torch.clamp(1 - 3 * sigmoids.std(), 0, 1)
    return threshold

@torch.no_grad()
def threshold_calc3(gamma, label_one_hot, pred_s, pred_t):
    threshold_s = threshold_calc(gamma, label_one_hot, pred_s, pred_t)
    threshold_t = threshold_calc2(gamma, label_one_hot, pred_s, pred_t)
    return (threshold_s + threshold_t) / 2

def ova_loss(output, label_onehot):
    label_ne = 1 - label_onehot
    #loss = torch.sum( - F.logsigmoid(output) * label_onehot, dim = 1).mean() + \
    #    torch.max(- F.logsigmoid( -output) * label_ne, dim = 1)[0].mean()
    loss = torch.sum( -F.logsigmoid(output) * label_onehot, dim = 1).mean() + \
        torch.sum( -F.logsigmoid(-output) * label_ne, dim = 1).mean()
    return loss

@torch.no_grad()
def entropic_partial_wasserstein(mu_s, mu_t, cost, b, reg = 0.01, iter_max = 1000):
    parallel_num = len(b)
    trans = torch.exp( - cost / reg).repeat(parallel_num, 1, 1) # num * ds * dt
    trans = trans / trans.sum(2).sum(1).view(parallel_num, 1, 1) * b.view(parallel_num, 1, 1)
    p, q = mu_s.repeat(parallel_num, 1), mu_t.repeat(parallel_num, 1)   # num * ds, num * dt
    for _ in range(iter_max):
        temp_p = torch.minimum(p / (1e-16 + trans.sum(2)), torch.ones_like(p))  # num * ds
        trans = temp_p.unsqueeze(2) * trans # diagonal matmul
        temp_q = torch.minimum(q / (1e-16 + trans.sum(1)), torch.ones_like(q))  # num * st
        trans = temp_q.unsqueeze(1) * trans # diagonal matmul
        trans = trans / trans.sum(2).sum(1).view(parallel_num, 1, 1) * b.view(parallel_num, 1, 1)
    return trans, (trans * cost.repeat(parallel_num, 1, 1)).sum(2).sum(1)
        

