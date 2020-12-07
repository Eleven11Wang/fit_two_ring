
import numpy as np
import matplotlib.pylab as plt
from sympy.solvers import solve
from sympy import Symbol
import os
import numpy as np
PI= np.pi
import random
import math
from PIL import Image
from PIL import ImageColor
import ring_functions
import find_parameter

def help_loss_function3_find_idx(loss_array, idx_ls):
    min_loss = np.inf
    idx_want = 0
    for n in range(len(loss_array)):
        # print(loss_array)
        if n not in idx_ls:
            if loss_array[n] < min_loss:
                idx_want = n
                min_loss = loss_array[n]
    return idx_want, min_loss


def loss_function_new(want1, mic_data1):
    sumLoss = 0
    idx_ls = []
    lx = list(range(len(mic_data1[0])))
    for n in range(2):
        x, y = want1[:, n]
        dx = mic_data1[0] - x
        dy = mic_data1[1] - y

        dv = np.vstack([dx, dy])
        loss_array = np.sum((dv * dv), axis=0)
        idx = help_loss_function3_find_idx(loss_array, idx_ls)
        idx_ls.append(idx)

    if idx_ls[1] == idx_ls[0] + 1 or (idx_ls[1] + 9 == idx_ls[0]):
        new_ls = lx[idx_ls[0]:] + lx[0:idx_ls[0]]
    else:
        new_ls = lx[0:idx_ls[0] + 1][::-1] + lx[idx_ls[0] + 1:][::-1]

    new_array = mic_data1[:, new_ls]
    d_array = want1 - new_array
    loss_array = np.sum((d_array * d_array), axis=0)

    return sum(loss_array)


def find_loss(center1, center2, parameter, rxy):
    c1, c2, h1, dh = parameter
    rx, ry = rxy
    rx = (rx * PI) / 180
    ry = (ry * PI) / 180
    rz = 0
    rotation_matrix = ring_functions.rotatexyz(rx, ry, rz)
    output_center1 = np.dot(rotation_matrix, np.array([c1, c2, h1]))
    # output_center1=output_center1[:2]
    output_center2 = np.dot(rotation_matrix, np.array([c1, c2, h1 + dh]))
    # output_center2=output_center2[:2]
    # print(output_center1,output_center2)
    loss = (output_center1[0] - center1[0]) ** 2 + (output_center1[1] - center1[1]) ** 2 \
           + (output_center2[0] - center2[0]) ** 2 + (output_center2[1] - center2[1]) ** 2
    return loss




def find_loss(center1, center2, parameter, rxy):
    c1, c2, h1, dh = parameter
    rx, ry = rxy
    rx = (rx * PI) / 180
    ry = (ry * PI) / 180
    rz = 0
    rotation_matrix = ring_functions.rotatexyz(rx, ry, rz)
    output_center1 = np.dot(rotation_matrix, np.array([c1, c2, h1]))
    # output_center1=output_center1[:2]
    output_center2 = np.dot(rotation_matrix, np.array([c1, c2, h1 + dh]))
    # output_center2=output_center2[:2]
    # print(output_center1,output_center2)
    loss = (output_center1[0] - center1[0]) ** 2 + (output_center1[1] - center1[1]) ** 2 \
           + (output_center2[0] - center2[0]) ** 2 + (output_center2[1] - center2[1]) ** 2
    return loss


def matrix_loss(rotated_mic_data,real_data):
    pin,idx_ls=find_parameter.find_pin(rotated_mic_data,real_data)
    loss_array=(rotated_mic_data[:2,idx_ls]-real_data)
    loss=sum(sum(loss_array*loss_array))
    return idx_ls,np.sqrt(loss)


def find_loss_position(parameter,rotate_matrix,real_position):
    localization=np.dot(rotate_matrix,np.array(parameter))
    real_loss=(localization[0]-real_position[0])**2+(localization[1]-real_position[1])**2
    #circle_loss=sum((localization-np.array(parameter))*(localization-np.array(parameter)))
    loss=real_loss
    return loss
