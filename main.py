import argparse
import yaml
import os
import importlib
from torch.utils.tensorboard.writer import SummaryWriter
from utils import create_dirs_if_not_exist, save_settings
import time
import easydict
import numpy as np
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str, default = './configs/UOT/visda_T2V_opda.yaml')
    parser.add_argument('--gpu', type = int, nargs = '+', default = [0])
    parser.add_argument('--seed', type = int, default = 1)
    opts, _ = parser.parse_known_args()
    log_dir = opts.config.split('configs/')[1].replace('/', '_')[:-5]
    time_now = time.strftime("_%Y-%m-%d_%H:%M:%S", time.localtime())
    parser.add_argument('--log_dir', type = str, default = os.path.join('results', log_dir + time_now))
    opts = parser.parse_args()
    config_yaml = yaml.load(open(opts.config), Loader = yaml.FullLoader)
    # merge the config from yaml and parser
    opts = easydict.EasyDict(d = dict(vars(opts), **config_yaml))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(id) for id in opts.gpu])
    torch.backends.cudnn.benchmark = True
    # set the random seed 
    os.environ['PYTHONHASHSEED'] = str(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.random.manual_seed(opts.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opts.seed) 
        torch.cuda.manual_seed_all(opts.seed)
    # create log folder
    create_dirs_if_not_exist(opts.log_dir)
    create_dirs_if_not_exist(os.path.join(opts.log_dir, 'figures'))
    # save settings
    save_settings(opts, opts.log_dir, vars(opts))
    writer = SummaryWriter(log_dir = opts.log_dir)
    # load method and run
    importlib.import_module('methods.{:}'.format(opts.method)).__getattribute__('main')(opts, writer)
    writer.close()
