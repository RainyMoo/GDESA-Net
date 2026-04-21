import torch.optim as Optim


# class WarmupOptimizer(object):
#     def __init__(self, lr_base, optimizer, data_size, batch_size, warmup_epoch):
#         self.optimizer = optimizer
#         self._step = 0
#         self.lr_base = lr_base
#         self._rate = 0
#         self.data_size = data_size
#         self.batch_size = batch_size
#         self.warmup_epoch = warmup_epoch


#     def step(self):
#         self._step += 1

#         rate = self.rate()
#         for p in self.optimizer.param_groups:
#             p['lr'] = rate
#         self._rate = rate

#         self.optimizer.step()


#     def zero_grad(self):
#         self.optimizer.zero_grad()


#     def rate(self, step=None):
#         if step is None:
#             step = self._step

#         if step <= int(self.data_size / self.batch_size * (self.warmup_epoch + 1) * 0.25):
#             r = self.lr_base * 1/(self.warmup_epoch + 1)
#         elif step <= int(self.data_size / self.batch_size * (self.warmup_epoch + 1) * 0.5):
#             r = self.lr_base * 2/(self.warmup_epoch + 1)
#         elif step <= int(self.data_size / self.batch_size * (self.warmup_epoch + 1) * 0.75):
#             r = self.lr_base * 3/(self.warmup_epoch + 1)
#         else:
#             r = self.lr_base

#         return r


# def get_optim(__C, model, data_size, lr_base=None):
#     if lr_base is None:
#         lr_base = __C.LR_BASE

#     std_optim = getattr(Optim, __C.OPT)
#     params = filter(lambda p: p.requires_grad, model.parameters())
#     eval_str = 'params, lr=0'
#     for key in __C.OPT_PARAMS:
#         eval_str += ' ,' + key + '=' + str(__C.OPT_PARAMS[key])

#     optim = WarmupOptimizer(
#         lr_base,
#         eval('std_optim' + '(' + eval_str + ')'),
#         data_size,
#         __C.BATCH_SIZE,
#         __C.WARMUP_EPOCH
#     )

#     return optim


# def adjust_lr(optim, decay_r):
#     optim.lr_base *= decay_r

class WarmupOptimizer(object):
    def __init__(self, optimizer, lr_bases, data_size, batch_size, warmup_epoch):
        self.optimizer = optimizer
        
        # 多组 base lr （对应 param_groups）
        self.lr_bases = lr_bases                  
        
        # 用第一个作为 lr_base（兼容旧框架）
        self.lr_base = lr_bases[0]               
        
        self.data_size = data_size
        self.batch_size = batch_size
        self.warmup_epoch = warmup_epoch

        self._step = 0
        self._rate = self.lr_base   # 兼容旧框架的打印用


    def step(self):
        self._step += 1
        rates = self.get_lrs()

        # 更新 param_groups 的 lr
        for g, lr in zip(self.optimizer.param_groups, rates):
            g['lr'] = lr

        # 保存“主 lr”方便打印与日志（兼容旧框架要求）
        self._rate = rates[0]

        self.optimizer.step()


    def zero_grad(self):
        self.optimizer.zero_grad()


    def get_lrs(self):
        """
        为 each param_group 计算 warmup 之后的真实 lr
        """
        step = self._step
        T = int(self.data_size / self.batch_size * (self.warmup_epoch + 1))

        if step <= 0.25 * T:
            scale = 1/(self.warmup_epoch + 1)
        elif step <= 0.50 * T:
            scale = 2/(self.warmup_epoch + 1)
        elif step <= 0.75 * T:
            scale = 3/(self.warmup_epoch + 1)
        else:
            scale = 1.0

        # 每个 param_group 独立 warmup
        return [lr_base * scale for lr_base in self.lr_bases]

def get_optim(__C, model, data_size):
    bert_params = []
    clip_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if "bert" in lname:
            bert_params.append(p)
        elif "clip" in lname:
            clip_params.append(p)
        else:
            other_params.append(p)

    lr_other = __C.LR_BASE
    lr_bert  = getattr(__C, "LR_BERT", 2e-5)
    lr_clip  = getattr(__C, "LR_CLIP", 2e-5)

    param_groups = []
    lr_bases = []

    # 普通网络参数
    if len(other_params) > 0:
        param_groups.append({"params": other_params, "lr": lr_other})
        lr_bases.append(lr_other)

    # BERT
    if len(bert_params) > 0:
        param_groups.append({"params": bert_params, "lr": lr_bert})
        lr_bases.append(lr_bert)

    # CLIP
    if len(clip_params) > 0:
        param_groups.append({"params": clip_params, "lr": lr_clip})
        lr_bases.append(lr_clip)

    optimizer = Optim.AdamW(
        param_groups, betas=(0.9,0.98), eps=1e-9
    )

    return WarmupOptimizer(
            optimizer,
            lr_bases,
            data_size,
            __C.BATCH_SIZE,
            __C.WARMUP_EPOCH
        )

def adjust_lr(optim, decay_r):
    optim.lr_base *= decay_r              # for logging + ckpt
    optim.lr_bases = [lr * decay_r for lr in optim.lr_bases]


#### lr_bert不变版本 ####
# class WarmupOptimizer(object):
#     def __init__(self, optimizer, lr_bases, data_size, batch_size, warmup_epoch, no_decay_idx=None):
#         self.optimizer = optimizer

#         self.lr_bases = lr_bases
#         self.lr_base = lr_bases[0]

#         self.no_decay_idx = no_decay_idx or []  # 哪些 group 不需要 decay

#         self.data_size = data_size
#         self.batch_size = batch_size
#         self.warmup_epoch = warmup_epoch

#         self._step = 0
#         self._rate = self.lr_base


#     def step(self):
#         self._step += 1
#         rates = self.get_lrs()

#         # 更新每组 param_group 的 lr
#         for i, g in enumerate(self.optimizer.param_groups):
#             g['lr'] = rates[i]

#         self._rate = rates[0]
#         self.optimizer.step()


#     def zero_grad(self):
#         self.optimizer.zero_grad()


#     def get_lrs(self):
#         step = self._step
#         T = int(self.data_size / self.batch_size * (self.warmup_epoch + 1))

#         # === warmup scale ===
#         if step <= 0.25 * T:
#             scale = 1/(self.warmup_epoch + 1)
#         elif step <= 0.50 * T:
#             scale = 2/(self.warmup_epoch + 1)
#         elif step <= 0.75 * T:
#             scale = 3/(self.warmup_epoch + 1)
#         else:
#             scale = 1.0

#         # === important: 为 no_decay groups 保持 lr 不变 ===
#         rates = []
#         for i, lr in enumerate(self.lr_bases):
#             if i in self.no_decay_idx:
#                 rates.append(lr)      # 固定学习率
#             else:
#                 rates.append(lr * scale)
#         return rates

# def get_optim(__C, model, data_size,lr_other=None):
#     bert_params = []
#     clip_params = []
#     other_params = []

#     for name, p in model.named_parameters():
#         if not p.requires_grad:
#             continue
#         lname = name.lower()
#         if "bert" in lname:
#             bert_params.append(p)
#         elif "clip" in lname:
#             clip_params.append(p)
#         else:
#             other_params.append(p)

#     if lr_other is not None:
#         lr_other = __C.LR_BASE
#     lr_bert  = getattr(__C, "LR_BERT", 2e-5)
#     lr_clip  = getattr(__C, "LR_CLIP", 2e-5)

#     param_groups = []
#     lr_bases = []
#     no_decay_idx = []

#     # 0: other params
#     param_groups.append({"params": other_params, "lr": lr_other})
#     lr_bases.append(lr_other)

#     # 1: BERT params（不 decay）
#     if len(bert_params) > 0:
#         param_groups.append({"params": bert_params, "lr": lr_bert})
#         lr_bases.append(lr_bert)
#         no_decay_idx.append(1)

#     # 2: CLIP params（不 decay）
#     if len(clip_params) > 0:
#         param_groups.append({"params": clip_params, "lr": lr_clip})
#         lr_bases.append(lr_clip)
#         no_decay_idx.append(2)

#     optimizer = Optim.AdamW(param_groups, betas=(0.9,0.98), eps=1e-9)

#     return WarmupOptimizer(
#         optimizer,
#         lr_bases,
#         data_size,
#         __C.BATCH_SIZE,
#         __C.WARMUP_EPOCH,
#         no_decay_idx=no_decay_idx
#     )

# def adjust_lr(optim, decay_r):
#     """
#     decay_r：衰减率，例如 0.2
#     只对非 no_decay_idx 的组进行 decay
#     """
#     # 主 lr_base 只用于日志，不用于真实学习率
#     optim.lr_base *= decay_r

#     new_lr_bases = []
#     for idx, lr in enumerate(optim.lr_bases):
#         if hasattr(optim, "no_decay_idx") and idx in optim.no_decay_idx:
#             # BERT / CLIP：不 decay
#             new_lr_bases.append(lr)
#         else:
#             # 普通组：进行 decay
#             new_lr_bases.append(lr * decay_r)

#     optim.lr_bases = new_lr_bases