# -*- coding: utf-8 -*-
# @Time    : 2019/5/16 9:38 PM
# @Author  : weiziyang
# @FileName: train.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
# @Time    : 2019/5/16 11:27 AM
# @Author  : weiziyang
# @FileName: train.py
# @Software: PyCharm
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

import data
import model


def run(nets, loader, iterations, train):
    if train:
        [net.train() for net in nets]
    else:
        [net.eval() for net in nets]
    optimizers = [torch.optim.Adam(net.parameters(), lr=0.01) for net in nets]

    loss_function = nn.CrossEntropyLoss()
    data = []
    losses_list = []
    tq = tqdm(loader, total=iterations, ncols=0, position=2, desc='train' if train else 'val')
    for i, (a, b, c) in enumerate(tq):
        if i >= iterations:
            break

        a = Variable(a, requires_grad=False)
        b = Variable(b.transpose(1, 2).contiguous(), requires_grad=False)
        c = Variable(c, requires_grad=False)

        pred_cs = [net(a, b) for net in nets]
        losses = [loss_function(pred_c, c) for pred_c in pred_cs]
        losses_list.append(losses)

        if train:
            [optimizer.zero_grad() for optimizer in optimizers]
            [loss.backward() for loss in losses]
            [optimizer.step() for optimizer in optimizers]
            data.append(extract_plins(nets[0]))
        else:
            acc = [(pred_c.data.max(dim=1)[1] == c.data).float().mean() for pred_c in pred_cs]
            data.append(acc)
    data = list(zip(*data))
    if not train:
        data = [np.mean([each.item() for each in d]) for d in data]
    else:
        torch.save(losses_list, 'loss.pth')
    return data


def main(objects, **kwargs):
    nets = [
        model.Net(objects),
        model.Baseline(objects),
    ]
    loader = get_loader(objects, **kwargs)
    plins = run(nets, loader, 1000, train=True)
    accs = run(nets, loader, 200, train=False)
    return {'plins': plins, 'accs': accs}


def extract_plins(net):
    plins = [net.counter.f[i] for i in range(8)]
    return [p.weight.data.cpu().clone() for p in plins]


def get_loader(*args, **kwargs):
    dataset = data.ToyTask(*args, **kwargs)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1024,
        num_workers=8,
        pin_memory=True,
    )
    return loader

resolution = 100 + 1

param_ranges = {
        'coord': torch.linspace(0.2, 1, resolution),
        'noise': torch.linspace(0, 1, resolution)
}

if __name__ == "__main__":
    params = {
        'objects': 10,
        'coord': 0.2,
        'noise': 0
    }
    logs = []
    param_ranges = torch.linspace(0, 1, 100)
    for noise in tqdm(param_ranges):
        params['noise'] = noise
        log = main(**params)
        log['config'] = params
        logs.append(log)
    torch.save(logs, 'object_10.pth')