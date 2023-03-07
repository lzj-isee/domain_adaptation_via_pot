import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import numpy as np
from collections import Counter

def tensorboard_log(datas, writer, step):
    for key, value in datas.items():
        writer.add_scalar(key, value, global_step = step)

def match_information(gamma, label_s, label_t):
    threshold = gamma.max() / 2
    loc_x, loc_y = np.where(gamma > threshold)
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

def inv_lr_scheduler(optimizer, iter_num, gamma = 10, power = 0.75, max_iter = 10000):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr_ratio = (1 + gamma * min(1.0, iter_num / max_iter)) ** ( - power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_ratio * param_group['init_lr']

def plot_tsne(feature_source, feature_target, label_source, label_target, save_path):
    num_source = len(label_source)
    combined_features = np.vstack([feature_source, feature_target])
    combined_labels = np.vstack([label_source, label_target]).astype('int')
    tsne = TSNE(perplexity = 30, n_components = 2, init = 'pca', n_iter = 3000)
    results = tsne.fit_transform(combined_features)
    plt.figure(figsize=(15,15))
    plt.scatter(results[:num_source, 0], results[:num_source, 1], c = combined_labels[0, :], s = 50, alpha = 0.5, marker = 'o', cmap = cm.jet, label = 'source')
    plt.scatter(results[num_source:, 0], results[num_source:, 1], c = combined_labels[1, :], s = 50, alpha = 0.5, marker = '+', cmap = cm.jet, label = 'target')
    plt.axis('off')
    plt.legend(loc = 'best')
    plt.savefig(save_path, bbox_inches = 'tight')
    plt.close()

def create_dirs_if_not_exist(dir_list):
    if isinstance(dir_list, list):
        for dir in dir_list:
            if not os.path.exists(dir):
                os.makedirs(dir)
    else:
        if not os.path.exists(dir_list):
            os.makedirs(dir_list)

def save_settings(opts, save_folder, settings):
    # save settings
    os.system('cp ./methods/{}.py {}/method.py'.format(opts.method, save_folder))
    if opts.method == 'UOT':
        os.system('cp ./networks/__init__.py {}/network.py'.format(save_folder))
    else:
        os.system('cp ./networks/{}.py {}/network.py'.format(opts.method, save_folder))
    with open(save_folder+'/settings.md',mode='w') as f:
        for key in settings:
            f.write(key + ': ' + '{}'.format(settings[key]) + ' \n')
            f.write('\n')

def remove_log(save_folder):
    # remove log files
    if os.path.exists(save_folder):
        names = os.listdir(save_folder)
        for name in names:
            os.remove(save_folder+'/'+name)
        print('remove files in {}'.format(save_folder))
    else:
        pass

def str2bool(v):
    return v.lower() in ('true')