

def inv_lr_scheduler(optimizer, iter_num, gamma = 10, power = 0.75, init_lr = 0.001, max_iter = 10000):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * min(1.0, iter_num / max_iter)) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr