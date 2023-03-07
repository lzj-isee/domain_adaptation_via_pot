from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F



def get_net_optim(opts):
    num_classes = opts.dataset.n_share + opts.dataset.n_source_private
    net = Full_net(pre_train = True, out_dim = num_classes)
    net.cuda()
    optimizer = optim.SGD(
        [
            {'params': net.res_feature.parameters(), 'lr': opts.lr_g, 'init_lr': opts.lr_g}, 
            {'params': net.fc1.parameters(), 'lr': opts.lr_c, 'init_lr': opts.lr_c}, 
            {'params': net.fc2.parameters(), 'lr': opts.lr_c, 'init_lr': opts.lr_c}
        ], weight_decay = opts.weight_decay, momentum = 0.9, nesterov = True
    )
    net = nn.DataParallel(net)
    return net, optimizer

class BaseFeatureExtractor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()

    def output_num(self):
        pass

    def train(self, mode=True):
        # freeze BN mean and std
        for module in self.children():
            if isinstance(module, nn.BatchNorm2d):
                module.train(False)
            else:
                module.train(mode)

class Full_net(BaseFeatureExtractor):
    def __init__(self, pre_train=True, out_dim = 12):
        super(Full_net, self).__init__()
        self.feature_dim = 2048
        self.bottle_dim = 512
        model_ft = models.resnet50(pretrained = pre_train)
        mod = list(model_ft.children())
        mod.pop()
        self.res_feature = nn.Sequential(
            *mod, 
            nn.Flatten(start_dim = 1)
        )
        self.fc1 = nn.Linear(self.feature_dim, self.bottle_dim)
        self.fc2 = nn.Linear(self.bottle_dim, out_dim)
        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)

    @torch.cuda.amp.autocast_mode.autocast()
    def forward(self, x, fc_only = False):
        if not fc_only: 
            x = self.res_feature(x)
            x = F.relu(self.fc1(x))
            y = self.fc2(x)
            return x, y
        else:
            return self.fc2(x)

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