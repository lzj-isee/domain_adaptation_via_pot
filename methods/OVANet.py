import torch
from dataloader import get_loaders
from networks import get_net_optim
from utils.OVANet import inv_lr_scheduler
import ot
import numpy as np
import torch.nn.functional as F

def ova_loss(output, label_onehot):
    output_softmax = F.softmax(output, dim = 1)
    label_ne = 1 - label_onehot
    loss = torch.sum(-torch.log(output_softmax[:, 1, :]+1e-8)*label_onehot, dim = 1).mean() + \
        torch.max(-torch.log(output_softmax[:, 0, :]+1e-8)*label_ne, dim = 1)[0].mean()
    return loss

def main(opts, writer):
    num_classes = opts.dataset.n_share + opts.dataset.n_source_private
    batch_size = opts.dataloader.batch_size
    # get data_loader
    source_loader, target_loader, test_loader = get_loaders(opts)
    # get model and optimizer
    net_g, net_c_n, net_c_ova, optim_g, optim_c = get_net_optim(opts)
    # train
    iter_s, iter_t = iter(source_loader), iter(target_loader)
    len_iter_s, len_iter_t = len(iter_s), len(iter_t)
    for step in range(opts.max_step):
        net_g.train()
        net_c_n.train()
        net_c_ova.train()
        if step % len_iter_s == 0: iter_s = iter(source_loader)
        if step % len_iter_t == 0: iter_t = iter(target_loader)
        data_s, data_t = next(iter_s), next(iter_t)
        inv_lr_scheduler(optim_g, iter_num = step, init_lr = opts.lr_g, max_iter = opts.max_step)
        inv_lr_scheduler(optim_c, iter_num = step, init_lr = opts.lr_c, max_iter = opts.max_step)
        image_s, label_s, image_t = data_s[0].cuda(), data_s[1].cuda(), data_t[0].cuda()
        net_c_ova.module.weight_norm()
        # forward
        feature_s, feature_t = net_g(image_s), net_g(image_t)
        out_n_s, out_ova_s, out_ova_t= net_c_n(feature_s), net_c_ova(feature_s), net_c_ova(feature_t)
        # classification loss on source
        loss_s = F.cross_entropy(out_n_s, label_s, reduction = 'mean')
        with torch.no_grad():
            label_source_one_hot = torch.zeros(batch_size, num_classes).cuda().scatter_(1, label_s.unsqueeze(1), 1)
        # open classifier loss on source
        loss_ova = ova_loss(out_ova_s.view(batch_size, 2, num_classes), label_source_one_hot) * 0.5
        # open classifier loss on target (do not use here)
        out_ova_t= F.softmax(out_ova_t.view(batch_size, 2, num_classes), dim = 1)
        ent_open = torch.mean(torch.mean(torch.sum(- out_ova_t * torch.log(out_ova_t + 1e-8), 1), 1)) * opts.lambda_
        loss = loss_s + loss_ova + ent_open
        # backward and step
        optim_g.zero_grad()
        optim_c.zero_grad()
        loss.backward()
        optim_g.step()
        optim_c.step()
        # print and test
        if step % opts.print_interval == 0:
            print('step:[{}/{}], loss_s:[{:.2e}], loss_ova:[{:.2e}], ent_open:[{:.2e}]'.format(
                step+1, opts.max_step, loss.item(), loss_ova.item(), ent_open.item()
            ))
        if step % opts.test_interval == 0:
            acc, h_score = test(opts, test_loader, net_g, net_c_n, net_c_ova)
            print('step:[{}/{}], acc:[{:.2f}], h_score:[{:.2f}]'.format(
                step+1, opts.max_step, acc, h_score
            ))

@torch.no_grad()
def test(opts, test_loader, net_g, net_c_n, net_c_ova):
    net_g.eval()
    net_c_n.eval()
    net_c_ova.eval()
    class_list = [i for i in range(opts.dataset.n_share)]
    class_list.append(opts.dataset.n_share + opts.dataset.n_source_private)
    correct_close, correct_all, correct_per_class = 0, 0, np.zeros(opts.dataset.n_share + 1)
    per_class_num, test_num = np.zeros_like(correct_per_class), 0
    for batch_idx, data in enumerate(test_loader):
        images, labels = data[0].cuda(), data[1].cuda()
        test_num += len(labels)
        features = net_g(images)
        out_n, out_ova = net_c_n(features), net_c_ova(features)
        pred = torch.max(out_n, dim = 1)[1]
        correct_close += (pred == labels).sum().cpu()
        out_ova = F.softmax(out_ova.view(out_ova.size(0), 2, -1), dim = 1)
        pred_unknow = out_ova[torch.arange(0, out_n.size(0)).long().cuda(), 0, pred]
        idx_unknow = np.where(pred_unknow.cpu().numpy() > 0.5)[0]
        pred[idx_unknow] = opts.dataset.n_share + opts.dataset.n_source_private
        correct_all += (pred == labels).sum().cpu()
        for i, class_ in enumerate(class_list):
            class_idx = np.where(labels.cpu().numpy() == class_)[0]
            correct_per_class[i] += len(np.where(pred[class_idx].cpu().numpy() == class_)[0])
            per_class_num[i] += len(class_idx)
    acc_all = correct_all / test_num * 100
    per_class_acc = correct_per_class / per_class_num * 100
    acc_close = correct_close / per_class_num[:-1].sum()
    avg_known_acc, unknown_acc = per_class_acc[:-1].mean(), per_class_acc[-1]
    h_score = 2 * avg_known_acc * unknown_acc / (avg_known_acc + unknown_acc)
    return acc_all, h_score


        









