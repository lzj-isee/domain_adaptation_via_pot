import os
import torchvision.transforms as transforms
from dataset import myImageFolder
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import copy


# note that the common class is the 10 classes shared by Office-31 and Caltech-256, not in the in alphabetical order
COMMON_CLASS_IN_OFFICE31 = [
    'back_pack', 'bike', 'calculator', 'headphones', 'keyboard', 
    'laptop_computer', 'monitor', 'mouse', 'mug', 'projector'
]

def pack_image_path(root_dir, opts, if_source):
    class_list = os.listdir(root_dir)
    class_list.sort()
    share_class, source_private_class, target_private_class = None, None, None
    if opts.dataset.name == 'Office31':
        if opts.dataset.n_share == 31:
            share_class = copy.deepcopy(class_list)
        else:
            share_class = COMMON_CLASS_IN_OFFICE31
        for class_ in share_class:
            class_list.remove(class_)
        source_private_class = class_list[0 : opts.dataset.n_source_private]
        target_private_class = class_list[opts.dataset.n_source_private : ]
    elif opts.dataset.name == 'OfficeHome':
        share_class = copy.deepcopy(class_list[0: opts.dataset.n_share])
        for class_ in share_class:
            class_list.remove(class_)
        source_private_class = class_list[0 : opts.dataset.n_source_private]
        target_private_class = class_list[opts.dataset.n_source_private : ]
    elif opts.dataset.name == 'VisDA':
        class_list.remove('image_list.txt')
        share_class = copy.deepcopy(class_list[0 : opts.dataset.n_share])
        for class_ in share_class:
            class_list.remove(class_)
        source_private_class = class_list[0 : opts.dataset.n_source_private]
        target_private_class = class_list[opts.dataset.n_source_private : ]
    elif opts.dataset.name == 'DomainNet':
        pass
    else:
        raise ValueError('wrong dataset name')
    source_class, target_class = share_class + source_private_class, share_class + target_private_class
    image_paths, labels = [], []
    if if_source:
        for class_idx, class_ in enumerate(source_class):
            image_listdir = os.listdir(os.path.join(root_dir, class_))
            image_paths += [os.path.join(root_dir, class_, image_path) for image_path in image_listdir]
            labels += [np.ones(len(image_listdir)) * class_idx]
    else:
        for class_idx, class_ in enumerate(target_class):
            image_listdir = os.listdir(os.path.join(root_dir, class_))
            image_paths += [os.path.join(root_dir, class_, image_path) for image_path in image_listdir]
            if class_idx < len(share_class):
                labels += [np.ones(len(image_listdir)) * class_idx]
            else:
                labels += [np.ones(len(image_listdir)) * len(source_class)]
    labels = np.concatenate(labels, axis = 0).astype('int64')
    return image_paths, labels

def get_loaders(opts):
    # define transforms
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # get the folder of images
    source_root_dir = os.path.join('../dataset', opts.dataset.name, opts.dataset.source_domain)
    target_root_dir = os.path.join('../dataset', opts.dataset.name, opts.dataset.target_domain)
    # pack the paths of images
    source_image_paths, source_labels = pack_image_path(source_root_dir, opts, if_source = True)
    target_image_paths, target_labels = pack_image_path(target_root_dir, opts, if_source = False)
    source_set = myImageFolder(source_image_paths, source_labels, train_transforms, None, if_return_idx = True)
    target_set = myImageFolder(target_image_paths, target_labels, train_transforms, None, if_return_idx = True)
    test_set = myImageFolder(target_image_paths, target_labels, test_transforms, None, if_return_idx = True)
    if opts.dataloader.class_balance:
        freq = Counter(source_set.labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_set.labels]
        sampler = WeightedRandomSampler(source_weights, len(source_set.labels))
        print("use balanced loader")
        source_loader = DataLoader(
            source_set, batch_size = opts.dataloader.batch_size,
            sampler = sampler, drop_last = True, num_workers = opts.dataloader.num_workers
        )
    else:
        source_loader = DataLoader(
            source_set, batch_size = opts.dataloader.batch_size,
            shuffle = True, drop_last = True, num_workers = opts.dataloader.num_workers
        )
    target_loader = DataLoader(
        target_set, batch_size = opts.dataloader.batch_size,
        shuffle = True, drop_last = True, num_workers = opts.dataloader.num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size = opts.dataloader.batch_size,
        shuffle = False, num_workers = opts.dataloader.num_workers
    )
    return source_loader, target_loader, test_loader

class balanced_source_loader(object):
    def __init__(self, opts) -> None:
        super().__init__()
        self.train_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.opts = opts
        # get the folder of images
        source_root_dir = os.path.join('../dataset', opts.dataset.name, opts.dataset.source_domain)
        # pack the paths of images
        self.source_image_paths, self.source_labels = pack_image_path(source_root_dir, opts, if_source = True)
        self.source_set = myImageFolder(self.source_image_paths, self.source_labels, self.train_transforms, None, if_return_idx = True)

    def get_balanced_loader(self):
        source_labels, label_index = self.source_labels, np.arange(0, len(self.source_labels), 1)
        #shuffle
        random_index = np.random.permutation(len(source_labels))
        source_labels, label_index = source_labels[random_index], label_index[random_index]
        n_class = len(np.unique(source_labels))
        new_index, min_num = [], len(label_index)
        for i in range(n_class):
            s_index = np.nonzero(source_labels == i)[0]
            min_num = len(s_index) if min_num > len(s_index) else min_num
            new_index.append(label_index[s_index])
        for i in range(n_class):
            new_index[i] = new_index[i][0: min_num]
        new_index = np.array(new_index).transpose(1, 0).reshape(-1)
        return DataLoader(
            self.source_set, sampler = new_index, 
            batch_size = self.opts.dataloader.batch_size, num_workers = self.opts.dataloader.num_workers, drop_last = True, shuffle = False
        )

