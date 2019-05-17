# -*- coding: utf-8 -*-
# @Time    : 2019/5/12 10:54 PM
# @Author  : weiziyang
# @FileName: visualization.py
# @Software: PyCharm

import data
import numpy as np
from graphviz import Digraph
import matplotlib.pyplot as plt
import matplotlib.patches as patches

cm = plt.cm.bwr


def visualize_graph(adjacent_matrix):
    """
    :param adjacent_matrix: A = a·a.T, a is attention weight
    :return: None
    """
    for i in range(adjacent_matrix.shape[0]):
        for j in range(adjacent_matrix.shape[1]):
            if adjacent_matrix[i][j] >= 1:
                adjacent_matrix[i][j] = 1
            else:
                adjacent_matrix[i][j] = 0

    size = len(adjacent_matrix)
    g = Digraph('Visualize_graph', filename='visualize_graph.gv', engine='circo')

    # create the nodes
    for i in range(size):
        content1 = adjacent_matrix[0][i]
        if content1 == 1:
            g.node('%d' %i, shape='circle', style='filled', fillcolor='black', fontcolor='white')
        else:
            g.node('%d' %i, shape='circle', style='filled', fillcolor='white', fontcolor='black')

    #create the edges
    for i in range(size):
        content1 = adjacent_matrix[0][i]
        content3 = adjacent_matrix[i][i]
        if (content3 == 1):
            g.edge('%d' %i, '%d' %i)

        for j in range(size):
            content2 = adjacent_matrix[i][j]
            if ((content2 == 1) and (j > i) ):
                g.edge('%d' %i, '%d' %j, dir='both')
    g.view()


def visualize_boxes(boxes_matrix, true_index):
    """
    :param boxes_matrix: 4 * N matrix
    :return:
    """
    plt.figure(figsize=(10, 10))
    axis = plt.gca()
    for index in range(boxes_matrix.size()[1]):
        coordinate = boxes_matrix[:, index]
        width = coordinate[2] - coordinate[0]
        height = coordinate[3] - coordinate[1]
        left_upper = (coordinate[0], coordinate[1])
        plt.text(coordinate[0]+width/2, coordinate[1]+height/2, str(index))
        if index in range(true_index):
            rect = patches.Rectangle(left_upper, width=width, height=height, linewidth=1, fill=True, alpha=0.7, color='r')
        else:
            rect = patches.Rectangle(left_upper, width=width, height=height, linewidth=1, fill=True, alpha=0.7, color='b')
        axis.add_patch(rect)
    plt.show()


def visualize_dataset():
    params = [0.1, 0.2, 0.3, 0.4, 0.5, None]
    plt.figure(figsize=(11.5, 4), dpi=200)
    for n, length in enumerate(params):
        task = data.Task(length=length, )
        weights, boxes, true_num = next(iter(task))
        print(true_num, len(weights))
        true_fig = plt.subplot(2, len(params), n*2+1, aspect='equal')
        data_fig = plt.subplot(2, len(params),  n*2+2, aspect='equal')

        for index in range(boxes.size()[1]):
            box = boxes[:, index]
            x = box[0]
            y = box[1]
            w = box[2] - box[0]
            h = box[3] - box[1]
            # Given images
            # As for color, "1" represents true, "0" represents false.
            # Therefore, the red rectangle represents true object.
            # the blue rectangle represents false object.
            data_fig.add_patch(patches.Rectangle((x, y), width=w, height=h, alpha=0.5,
                                                 linewidth=0, color=cm(float(weights[index]))))
            # In fact
            if index < true_num:
                true_fig.add_patch(patches.Rectangle((x, y), width=w, height=h, alpha=0.5, linewidth=0,
                                                     color=cm(float(1))))
            else:
                true_fig.add_patch(patches.Rectangle((x, y), width=w, height=h, alpha=0.5, linewidth=0,
                                                     color=cm(float(0))))

        true_fig.axes.get_xaxis().set_visible(False)
        data_fig.axes.get_xaxis().set_visible(False)
        true_fig.axes.get_yaxis().set_major_locator(plt.NullLocator())
        data_fig.axes.get_yaxis().set_visible(False)
        true_fig.set_title('Ground truth: {}'.format(true_num))
        data_fig.set_title('Given Data')
        if length:
            true_fig.set_ylabel('$length = {}$'.format(length))
        else:
            true_fig.set_ylabel('$length=random$')

    plt.show()

