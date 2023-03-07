import torch
from dataloader import get_loaders, balanced_source_loader
from networks import get_net_optim
from utils import inv_lr_scheduler ,tensorboard_log
import ot
import os
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from methods import ova_loss, threshold_calc3, entropic_partial_wasserstein, match_information, find_match, update_prototype
import matplotlib.pyplot as plt

def main(opts, writer):
    print(vars(opts))
    train(opts, writer)

def train(opts, writer):
    num_classes, batch_size = opts.dataset.n_share + opts.dataset.n_source_private, opts.dataloader.batch_size
    # get data_loader
    _, target_loader, test_loader = get_loaders(opts)
    source_loader = balanced_source_loader(opts)
    # get optimizer
    net, optimizer = get_net_optim(opts)
    scaler = torch.cuda.amp.GradScaler()
    # train
    iter_s, iter_t = iter(source_loader.get_balanced_loader()), iter(target_loader)
    len_iter_s, len_iter_t = len(iter_s), len(iter_t)
    prototype = torch.zeros((num_classes + 1, net.module.bottle_dim), device = 'cuda')
    temp = net.module.temp
    for step in range(opts.max_step):
        net.train()
        if (step % len_iter_s) == 0: iter_s = iter(source_loader.get_balanced_loader())
        if (step % len_iter_t) == 0: iter_t = iter(target_loader)
        data_s, data_t = next(iter_s), next(iter_t)
        inv_lr_scheduler(optimizer, iter_num = step, max_iter = opts.max_step)
        image_s, label_s, image_t = data_s[0].cuda(non_blocking=True), data_s[1].cuda(non_blocking=True), data_t[0].cuda(non_blocking=True)
        with torch.no_grad(): label_source_one_hot = torch.zeros(batch_size, num_classes, device = 'cuda').scatter_(1, label_s.unsqueeze(1), 1)
        eff = 1 - np.exp( - step / 500)
        alpha, lambda_t, beta = opts.alpha * eff, opts.lambda_t * eff, opts.beta * eff
        with torch.cuda.amp.autocast_mode.autocast():
            # forward
            feature_s, out_s = net(image_s)
            feature_t, out_t = net(image_t)
            pred_s, pred_t = torch.sigmoid(out_s), torch.sigmoid(out_t)
            # calculate the distance matrix
            with torch.no_grad():
                feature_s_n, feature_t_n = F.normalize(feature_s.detach()), F.normalize(feature_t.detach())
                prototype_n = F.normalize(prototype)
                cos_dissimi_st = 1 - torch.matmul(feature_s_n, feature_t_n.t())
                cos_simi_ct = torch.matmul(prototype_n, feature_t_n.t()) / temp
                weight_matrix = torch.matmul(label_source_one_hot, F.softmax( -cos_simi_ct, dim = 0)[:-1,:])
                cost_matrix = (cos_dissimi_st * (1 - weight_matrix)).cpu().numpy()
                cost_matrix = (cost_matrix - cost_matrix.min()) / (cost_matrix.max() - cost_matrix.min())
            # find param b, transition mass
            param_b_list = np.linspace(0.02, 1.0, 50)
            ot_list, gamma_list = [], []
            for value in param_b_list:
                gamma = ot.partial.partial_wasserstein(ot.unif(batch_size), ot.unif(batch_size), cost_matrix, m = value)
                ot_list.append((gamma * cost_matrix).sum())
                gamma_list.append(gamma)
            values = np.array(ot_list)
            values_n = (values - values.min()) / (values.max() - values.min())
            slope_list = (values_n[1 : len(values_n)] - values_n[0 : len(values_n) - 1]) * len(values_n)
            for k in range(len(slope_list)):
                if slope_list[k] > 0.8: break
            if step == 0: 
                param_b = k / len(values_n) 
            else: 
                param_b = k / len(values_n) * 0.1 + 0.9 * param_b
            # calculate optimal transport
            gamma = torch.from_numpy(gamma_list[int(param_b * len(values_n))]).float().cuda(non_blocking=True)
            # update threshold
            with torch.no_grad():
                net.module.update_threshold(threshold_calc3(gamma, label_source_one_hot, pred_s.detach(), pred_t.detach()))
            # loss
            threshold = net.module.threshold
            gamma_01 = torch.where(gamma > gamma.max() / 2, 1, 0)
            # loc_x, loc_y_match = torch.nonzero(gamma_01, as_tuple = True)
            loc_y_unmatch = torch.nonzero(1 - gamma_01.sum(0)).view(-1)
            # update prototype
            prototype = update_prototype(prototype, feature_s.detach(), label_source_one_hot, feature_t.detach(), loc_y_unmatch)
            loss_c = ova_loss(out_s, label_source_one_hot)
            # max_pred_t_value, max_pred_t_index = torch.max(pred_t, dim = 1)
            c0 = feature_s.pow(2).sum(1).view(-1, 1) + feature_t.pow(2).sum(1).view(1, -1) - 2.0 * torch.matmul(feature_s, feature_t.t())
            c1 = - torch.matmul(label_source_one_hot, F.logsigmoid(out_t).t()) - \
                torch.sum((1 - label_source_one_hot).unsqueeze(1).expand(batch_size, batch_size, num_classes) * \
                F.logsigmoid(-out_t).unsqueeze(0).expand(batch_size, batch_size, num_classes), dim = 2)
            loss_matrix = c1 * beta + c0 * alpha
            loss_align = (gamma * loss_matrix).sum() / gamma.sum()
            # idx_unknow = torch.nonzero(max_pred_t_value < threshold).view(-1)
            # loss_t = torch.mean(threshold * torch.log(pred_t + 1e-3) + (1 - threshold) * torch.log(1 - pred_t + 1e-3))
            #loss_un = - torch.min(F.logsigmoid(-out_t), dim = 1)[0].matmul(unmatch / unmatch.sum())
            p = F.softmax(torch.matmul(F.normalize(feature_t), F.normalize(prototype).t()) / temp, dim = 1)
            loss_ent = - torch.sum(p * torch.log(p + 1e-5), dim = 1).mean()
            loss = loss_c + loss_align + loss_ent * lambda_t# + loss_t * lambda_t# + loss_un * opts.beta
        # backward and step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # print and test
        with torch.no_grad():
            if ((step + 1) % opts.print_interval) == 0 or step == 0:
                # calculate matching information
                match_acc, true_match, match_rate = match_information(gamma, label_s.cpu().numpy(), data_t[1].cpu().numpy())
                print('step:[{}/{}], c:[{:.2e}], align:[{:.2e}], ent:[{:.2e}], b:[{:.2f}], inf:[{:.1f}, {:.1f}, {:.1f}], Thr:[{:.2f}]'.format(
                    step+1, opts.max_step, loss_c.item(), loss_align.item(), loss_ent.item(), 
                    param_b, match_acc, true_match, match_rate, threshold.item()
                ))
                # log training information
                tensorboard_log(
                    {
                        'b': param_b, 'm_acc': match_acc, 
                        'loss_c': loss_c.item(), 'loss_align': loss_align.item(), 'loss_ent': loss_ent.item(), 
                        'c0': (gamma * c0).sum().item(), 'c1': (gamma * c1).sum().item(), 'threshold': net.module.threshold.item()
                    }, writer, step
                )
                # save figures
                figure = plt.figure(figsize = (6.4, 4.8))
                plt.plot(param_b_list, values_n, color = 'r')
                plt.plot(param_b_list, param_b_list, color = 'b')
                plt.vlines(param_b, 0, 1, colors='g')
                plt.savefig(os.path.join(opts.log_dir, 'figures', '{:}.jpg'.format(step + 1)))
                plt.close()
            if ((step + 1) % opts.test_interval) == 0 or step == 0:
                acc_close, acc_all, h_score = test(opts, test_loader, net)
                print('step:[{}/{}], acc_close:[{:.2f}] acc_all:[{:.2f}], h_score:[{:.2f}]'.format(
                    step+1, opts.max_step, acc_close, acc_all, h_score
                ))
                tensorboard_log({
                    'acc_close': acc_close, 'acc_all': acc_all, 'h_score': h_score}, 
                    writer, step + 1
                )

@torch.no_grad()
def test(opts, test_loader, net):
    net.eval()
    num_classes = opts.dataset.n_share + opts.dataset.n_source_private
    class_list = [i for i in range(opts.dataset.n_share)]
    class_list.append(num_classes)
    correct_close, correct_all, correct_per_class = 0, 0, np.zeros(opts.dataset.n_share + 1)
    per_class_num, test_num = np.zeros_like(correct_per_class), 0
    threshold = net.module.threshold.cpu().numpy()
    for batch_idx, data in enumerate(test_loader):
        images, labels = data[0].cuda(), data[1].cuda()
        test_num += len(labels)
        _, out = net(images)
        pred = torch.max(out, dim = 1)[1]
        correct_close += (pred == labels).sum().cpu()
        out_sigmoid = torch.sigmoid(out)
        pred_sigmoid = out_sigmoid[torch.arange(0, out.size(0)).long().cuda(), pred]
        idx_unknow = np.where(pred_sigmoid.cpu().numpy() < threshold)[0]
        pred[idx_unknow] = num_classes
        correct_all += (pred == labels).sum().cpu()
        for i, class_ in enumerate(class_list):
            class_idx = np.where(labels.cpu().numpy() == class_)[0]
            correct_per_class[i] += len(np.where(pred[class_idx].cpu().numpy() == class_)[0])
            per_class_num[i] += len(class_idx)
    acc_all = correct_all / test_num * 100
    #per_class_acc = correct_per_class / per_class_num * 100
    per_class_acc = np.zeros_like(correct_per_class)
    for i in range(len(per_class_acc)): 
        if per_class_num[i] == 0: 
            per_class_acc[i] = 0
        else:
            per_class_acc[i] = correct_per_class[i] / per_class_num[i] * 100
    acc_close = correct_close / per_class_num[:-1].sum() * 100
    avg_known_acc, unknown_acc = per_class_acc[:-1].mean(), per_class_acc[-1]
    h_score = 2 * avg_known_acc * unknown_acc / (avg_known_acc + unknown_acc)
    return acc_close, acc_all, h_score



        









