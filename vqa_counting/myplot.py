# -*- coding: utf-8 -*-
# @Time    : 2019/5/16 11:57 AM
# @Author  : weiziyang
# @FileName: myplot.py
# @Software: PyCharm
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt


def real_situation():
    data = torch.load('real_situation.pth')
    count_acc = []
    base_acc = []
    noise = []
    for each in data:
        count, base = each['accs']
        count_acc.append(count)
        base_acc.append(base)
        noise.append(each['noise'])
    plt.figure(figsize=(5, 5))
    plt.plot(noise, count_acc, label='counting')
    plt.plot(noise, base_acc, label='base')
    plt.xlabel('noise')
    plt.ylabel('accuracy')
    plt.title('Accuracy of Random Object length')
    plt.legend()
    plt.show()


def loss_figure():
    data = torch.load('loss.pth')
    count_loss_list = []
    base_loss_list = []
    for each in data:
        count_loss, base_loss = [_.data for _ in each]
        count_loss_list.append(float(count_loss))
        base_loss_list.append(float(base_loss))
    x = torch.linspace(0, 1000, 1000)
    plt.plot([float(each) for each in x], count_loss_list, label='count')
    plt.plot([float(each) for each in x], base_loss_list, label='base_line')
    plt.xlabel('Iteration(n)')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('$Loss\ curve(noise=0, object\_length=0.2, object=10)$')
    plt.show()


def f32():
    with open('f32.pkl', 'rb') as f:
        data = pickle.load(f)
        x = data['coord']
        y1 = data['accs_count']
        y2 = data['accs_base_line']
        plt.xlabel('length of objects')
        plt.ylabel('accuracy')
        plt.title('The accuracy change with the object length')
        plt.plot(x, y1, label='count')
        plt.plot(x, y2, label='base_line')
        plt.legend()
        plt.show()


def confidence():
    data = torch.load('confidence01.pth')
    count_lost_list = []
    base_loss_list = []
    for each in data:
        count, base = each['accs']
        count_lost_list.append(count)
        base_loss_list.append(base)
    x = np.linspace(0, 1, 10)
    plt.figure(figsize=(5, 5))
    plt.plot(x, count_lost_list, label='count')
    plt.plot(x, base_loss_list, label='base')
    plt.title('$Accuracy \ when\ confidence = 0.1$')
    plt.xlabel('Noise')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def objects30():
    data = torch.load('objects30.pth')
    count_lost_list = []
    base_loss_list = []
    for each in data:
        count, base = each['accs']
        count_lost_list.append(count)
        base_loss_list.append(base)
    x = np.linspace(0, 1, 20)
    plt.figure(figsize=(5, 5))
    plt.plot(x, count_lost_list, label='count')
    plt.plot(x, base_loss_list, label='base')
    plt.title('$Accuracy \ when\ objects\_num=30$')
    plt.xlabel('Noise')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    objects30()