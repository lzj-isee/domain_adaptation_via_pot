import torch
from dataloader import get_loaders
from networks.CDAN import get_net, get_optim
from utils.CDAN import CDAN
from utils import inv_lr_scheduler
import ot
import os
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from methods import tensorboard_log


def main(opts, writer):
    num_classes = opts.dataset.n_share + opts.dataset.n_source_private
    batch_size = opts.dataloader.batch_size
    # get data_loader
    source_loader, target_loader, test_loader = get_loaders(opts)
    # get model and optimizer
    base_net, ad_net = get_net(opts)
    optimizer = get_optim(opts, base_net, ad_net, opts.lr)
    # train
    iter_s, iter_t = iter(source_loader), iter(target_loader)
    len_iter_s, len_iter_t = len(iter_s), len(iter_t)
    for step in range(opts.max_step):
        base_net.train()
        ad_net.train()
        if step % len_iter_s == 0: iter_s = iter(source_loader)
        if step % len_iter_t == 0: iter_t = iter(target_loader)
        data_s, data_t = next(iter_s), next(iter_t)
        image_s, label_s, image_t = data_s[0].cuda(), data_s[1].cuda(), data_t[0].cuda()
        feature_s, output_s = base_net(image_s)
        feature_t, output_t = base_net(image_t)
        features = torch.cat([feature_s, feature_t], dim = 0)
        outputs = torch.cat([output_s, output_t], dim = 0)
        softmax_out = F.softmax(outputs, dim = 1)
        loss_c = F.cross_entropy(output_s, label_s, reduction = 'mean')
        # CDAN without random
        transfer_loss = CDAN([features, softmax_out], ad_net, None, None, None)
        total_loss = opts.trade_off * transfer_loss + loss_c
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        # print and test
        if ((step + 1) % opts.print_interval) == 0 or step == 0:
            print('step:[{}/{}], loss_c:[{:.2e}], transfer_loss:[{:.2e}]'.format(
                step+1, opts.max_step, loss_c.item(), transfer_loss.item()
            ))
        if ((step + 1) % opts.test_interval) == 0 or step == 0:
            acc_close, acc, h_score = test(opts, test_loader, base_net)
            print('step:[{}/{}], acc_close:[{:.2f}] acc:[{:.2f}], h_score:[{:.2f}]'.format(
                step+1, opts.max_step, acc_close, acc, h_score
            ))

@torch.no_grad()
def test(opts, test_loader, base_net):
    base_net.eval()
    if opts.dataset.n_total - opts.dataset.n_source_private - opts.dataset.n_share > 0:
        raise ValueError('CDAN is only available on CDA and PDA')
        num_classes = opts.dataset.n_share + opts.dataset.n_source_private
        class_list = [i for i in range(opts.dataset.n_share)]
        class_list.append(opts.dataset.n_share + opts.dataset.n_source_private)
        correct_close, correct_all, correct_per_class = 0, 0, np.zeros(opts.dataset.n_share + 1)
        per_class_num, test_num = np.zeros_like(correct_per_class), 0
        for batch_idx, data in enumerate(test_loader):
            images, labels = data[0].cuda(), data[1].cuda()
            test_num += len(labels)
            features = net_g(images)
            out_n = net_c(features)
            pred = torch.max(out_n, dim = 1)[1]
            pred_wo_unknow = torch.where(pred.clone() == num_classes, torch.ones_like(pred).cuda().long() * (num_classes + 1), pred.clone())
            correct_close += (pred_wo_unknow == labels).sum().cpu()
            #out_ova = F.softmax(out_ova.view(out_ova.size(0), 2, -1), dim = 1)
            #pred_unknow = out_ova[torch.arange(0, out_n.size(0)).long().cuda(), 0, pred]
            #idx_unknow = np.where(pred_unknow.cpu().numpy() > 0.5)[0]
            #pred[idx_unknow] = opts.dataset.n_share + opts.dataset.n_source_private
            correct_all += (pred == labels).sum().cpu()
            for i, class_ in enumerate(class_list):
                class_idx = np.where(labels.cpu().numpy() == class_)[0]
                correct_per_class[i] += len(np.where(pred[class_idx].cpu().numpy() == class_)[0])
                per_class_num[i] += len(class_idx)
        acc_all = correct_all / test_num * 100
        per_class_acc = correct_per_class / per_class_num * 100
        acc_close = correct_close / per_class_num[:-1].sum() * 100
        avg_known_acc, unknown_acc = per_class_acc[:-1].mean(), per_class_acc[-1]
        h_score = 2 * avg_known_acc * unknown_acc / (avg_known_acc + unknown_acc)
        return acc_close, acc_all, h_score
    else:
        correct = 0
        test_num = 0
        for batch_idx, data in enumerate(test_loader):
            images, labels = data[0].cuda(), data[1].cuda()
            test_num += len(labels)
            _, outputs = base_net(images)
            pred = torch.max(outputs, dim = 1)[1]
            #out_ova = F.softmax(out_ova.view(out_ova.size(0), 2, -1), dim = 1)
            #pred_unknow = out_ova[torch.arange(0, out_n.size(0)).long().cuda(), 0, pred]
            #idx_unknow = np.where(pred_unknow.cpu().numpy() > 0.5)[0]
            #pred[idx_unknow] = opts.dataset.n_share + opts.dataset.n_source_private
            correct += (pred == labels).sum().cpu()
        acc = correct / test_num * 100
        return 0, acc, 0