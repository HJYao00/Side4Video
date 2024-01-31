from torch.optim.lr_scheduler import _LRScheduler
import math

def to_tuple(x, L):
    if type(x) in (int, float):
        return [x] * L
    if type(x) in (list, tuple):
        if len(x) != L:
            raise ValueError('length of {} ({}) != {}'.format(x, len(x), L))
        return tuple(x)
    raise ValueError('input {} has unkown type {}'.format(x, type(x)))

class WarmupLR(_LRScheduler):

    def __init__(self,
                 optimizer,
                 warmup_epochs=0,
                 warmup_powers=1,
                 warmup_lrs=0,
                 last_epoch=-1):
        self.num_groups = len(optimizer.param_groups)
        self.warmup_epochs = to_tuple(warmup_epochs, self.num_groups)
        self.warmup_powers = to_tuple(warmup_powers, self.num_groups)
        self.warmup_lrs = to_tuple(warmup_lrs, self.num_groups)
        super(WarmupLR, self).__init__(optimizer, last_epoch)
        assert self.num_groups == len(self.base_lrs)

    def get_lr(self):
        curr_lrs = []
        for group_index in range(self.num_groups):
            if self.last_epoch < self.warmup_epochs[group_index]:
                progress = self.last_epoch / self.warmup_epochs[group_index]
                factor = progress ** self.warmup_powers[group_index]
                lr_gap = self.base_lrs[group_index] - self.warmup_lrs[group_index]
                curr_lrs.append(factor * lr_gap + self.warmup_lrs[group_index])
            else:
                curr_lrs.append(self.get_single_lr_after_warmup(group_index))
        return curr_lrs

class WarmupCosineAnnealingLR(WarmupLR):

    def __init__(self,
                 optimizer,
                 total_epoch,
                 final_factor=0,
                 warmup_epochs=0,
                 warmup_powers=1,
                 warmup_lrs=0,
                 last_epoch=-1):
        self.total_epoch = total_epoch
        self.final_factor = final_factor
        super(WarmupCosineAnnealingLR, self).__init__(optimizer,
                                                      warmup_epochs,
                                                      warmup_powers,
                                                      warmup_lrs,
                                                      last_epoch)

    def get_single_lr_after_warmup(self, group_index):
        warmup_epoch = self.warmup_epochs[group_index]
        progress = (self.last_epoch - warmup_epoch) / (self.total_epoch - warmup_epoch)
        progress = min(progress, 1.0)
        cosine_progress = (math.cos(math.pi * progress) + 1) / 2
        factor = cosine_progress * (1 - self.final_factor) + self.final_factor
        return self.base_lrs[group_index] * factor

def _lr_scheduler(args, optimizer):
    lr_scheduler = WarmupCosineAnnealingLR(
            optimizer,
            total_epoch=args.epochs,
            warmup_epochs=args.warmup_proportion,
        )
    return lr_scheduler
