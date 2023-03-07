import torch
from dataloader import get_loaders, balanced_source_loader
from networks.PJDOT import get_net_optim
from utils import inv_lr_scheduler
from utils.PJDOT import match_information
import ot
import os
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from methods import tensorboard_log

def main(opts, writer):
    # get model
    print(vars(opts))
    train(opts, writer)

def train(opts, writer):
    num_classes = opts.dataset.n_share + opts.dataset.n_source_private
    batch_size = opts.dataloader.batch_size
    # get data_loader
    _, target_loader, test_loader = get_loaders(opts)
    source_loader = balanced_source_loader(opts)
    # get optimizer
    net, optimizer = get_net_optim(opts)
    scaler = torch.cuda.amp.GradScaler()
    # train
    iter_s, iter_t = iter(source_loader.get_balanced_loader()), iter(target_loader)
    len_iter_s, len_iter_t = len(iter_s), len(iter_t)
    for step in range(opts.max_step):
        net.train()
        if (step % len_iter_s) == 0: iter_s = iter(source_loader.get_balanced_loader())
        if (step % len_iter_t) == 0: iter_t = iter(target_loader)
        data_s, data_t = next(iter_s), next(iter_t)
        inv_lr_scheduler(optimizer, iter_num = step, max_iter = opts.max_step)
        image_s, label_s, image_t = data_s[0].cuda(non_blocking=True), data_s[1].cuda(non_blocking=True), data_t[0].cuda(non_blocking=True)
        param_b = opts.param_b
        with torch.no_grad(): label_source_one_hot = torch.zeros(batch_size, num_classes, device = 'cuda').scatter_(1, label_s.unsqueeze(1), 1)
        with torch.cuda.amp.autocast_mode.autocast():
            # forward
            feature_s, out_s = net(image_s)
            feature_t, out_t = net(image_t)
            # prediction on target
            pred_t = F.softmax(out_t, dim = 1)
            # calculate the distance matrix
            c0 = feature_s.pow(2).sum(1).view(-1, 1) + feature_t.pow(2).sum(1).view(1, -1) - 2.0 * torch.matmul(feature_s, feature_t.t())
            c1 = - torch.matmul(label_source_one_hot, torch.log(pred_t + 1e-8).t())
            dis_matrix = c0 * opts.alpha + c1 * opts.beta
            # dis_matrix = c1 * opts.beta
            gamma_np = ot.partial.partial_wasserstein(ot.unif(batch_size), ot.unif(batch_size), dis_matrix.detach().cpu().numpy(), m = param_b)
            gamma = torch.from_numpy(gamma_np).cuda(non_blocking=True) / param_b
            loss_c = F.cross_entropy(out_s, label_s, reduction = 'mean')
            loss_align = (gamma * dis_matrix).sum()
            loss = loss_c + loss_align
        # backward and step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # print and test
        with torch.no_grad():
            if ((step + 1) % opts.print_interval) == 0 or step == 0:
                match_acc, true_match, match_rate = match_information(gamma_np, label_s.cpu().numpy(), data_t[1].cpu().numpy())
                print('step:[{}/{}], c:[{:.2e}], align:[{:.2e}], b:[{:.2f}], inf:[{:.1f}, {:.1f}, {:.1f}]'.format(
                    step+1, opts.max_step, loss_c.item(), loss_align.item(), param_b, match_acc, true_match, match_rate
                ))
                tensorboard_log(
                    {
                        'b': param_b, 'm_acc': match_acc, 'loss_c': loss_c.item(), 'loss_align': loss_align.item(), 
                        'c0': (gamma * c0).sum().item(), 'c1': (gamma * c1).sum().item()
                    }, writer, step
                )
            if ((step + 1) % opts.test_interval) == 0 or step == 0:
                acc_close, acc, h_score = test(opts, test_loader, net)
                print('step:[{}/{}], acc_close:[{:.2f}] acc:[{:.2f}], h_score:[{:.2f}]'.format(
                    step+1, opts.max_step, acc_close, acc, h_score
                ))
                tensorboard_log({
                    'acc_close': acc_close, 'acc': acc, 'h_score': h_score}, 
                    writer, step + 1
                )

@torch.no_grad()
def test(opts, test_loader, net):
    net.eval()
    if opts.dataset.n_total - opts.dataset.n_source_private - opts.dataset.n_share > 0:
        raise ValueError('PJDOT is only available on CDA and PDA')
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
            _, out_n = net(images)
            pred = torch.max(out_n, dim = 1)[1]
            #out_ova = F.softmax(out_ova.view(out_ova.size(0), 2, -1), dim = 1)
            #pred_unknow = out_ova[torch.arange(0, out_n.size(0)).long().cuda(), 0, pred]
            #idx_unknow = np.where(pred_unknow.cpu().numpy() > 0.5)[0]
            #pred[idx_unknow] = opts.dataset.n_share + opts.dataset.n_source_private
            correct += (pred == labels).sum().cpu()
        acc = correct / test_num * 100
        return 0, acc, 0
