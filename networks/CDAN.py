from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import numpy as np
#from networks import Res50Feature, Classifier


def get_net(opts):
    num_classes = opts.dataset.n_share + opts.dataset.n_source_private
    base_net = Res50(pre_train = True, out_dim = num_classes)
    ad_net = AdversarialNetwork(in_feature = 256 * num_classes, hidden_size = 1024)
    base_net.cuda()
    ad_net.cuda()
    base_net = nn.DataParallel(base_net)
    ad_net = nn.DataParallel(ad_net)
    return base_net, ad_net

def get_optim(opts, base_net, ad_net, lr):
    optimizer = optim.SGD(
        [{'params': base_net.parameters()}, {'params': ad_net.parameters()}],
        lr = lr, momentum = 0.9, weight_decay = opts.weight_decay, nesterov = True
    )
    return optimizer

class Res50(nn.Module):
    def __init__(self, pre_train=True, out_dim = 12):
        super(Res50, self).__init__()
        self.out_dim = 2048
        model_ft = models.resnet50(pretrained = pre_train)
        mod = list(model_ft.children())
        mod.pop()
        self.features = nn.Sequential(
            *mod, 
            nn.Flatten(start_dim = 1),
        )
        self.bottleneck = nn.Linear(self.out_dim, 256)
        self.bottleneck.apply(init_weights)
        self.fc = nn.Linear(256, out_dim)
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.features(x)
        x = self.bottleneck(x)
        y = self.fc(x)
        return x, y

class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = 10000.0

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)
    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1