import numpy as np
PI=np.pi
import random_walk_functions
import ring_functions
import loss_functions



def find_mean_normal(normal1, normal2, angle_dict):
    possible_ls1 = []
    possible_ls2 = []
    mean_normal = []
    for idx in range(len(normal1)):
        mean_normal.append((normal1[idx] + normal2[idx]) / 2)

    ratio_normal = mean_normal[3] / mean_normal[2]

    for k, v in angle_dict.items():

        if abs((v[0] / v[1]) - ratio_normal) < 0.1:

            if abs(v[2] - mean_normal[4]) < 0.1:
                possible_ls1.append(k)
            if abs(v[3] - mean_normal[4]) < 0.1:
                possible_ls2.append(k)
    possible_ls = [possible_ls1, possible_ls2]
    return possible_ls


def find_normal(normal, angle_dict):
    possible_ls1 = []
    possible_ls2 = []

    if normal[2] < normal[3]:
        ratio = normal[3] / normal[2]

    else:
        ratio = normal[2] / normal[3]

    for k, v in angle_dict.items():

        if abs((v[0] / v[1]) - ratio) < 0.1:

            if abs(v[2] - normal[4]) < 0.1:
                possible_ls1.append(k)
            if abs(v[3] - normal[4]) < 0.1:
                possible_ls2.append(k)
    possible_ls = [possible_ls1, possible_ls2]
    return possible_ls


def find_possible_ls(normal1, normal2, angle_dict):
    d1 = max(normal1[2], normal1[3]) / 2
    d2 = max(normal2[2], normal2[3]) / 2

    if max(d1, d2) > 2 * min(d1, d2):
        possible_ls_mean = []
    else:
        possible_ls_mean = find_mean_normal(normal1, normal2, angle_dict)

    if d2 > d1:
        normal = normal2
    else:
        normal = normal1
    possible_ls_normal = find_normal(normal, angle_dict)
    possible_ls = [possible_ls_mean, possible_ls_normal]
    return possible_ls


def find_which_possible(normal1, normal2, angle_dict, real_data1, real_data2):
    d1 = max(normal1[2], normal1[3]) / 2
    d2 = max(normal2[2], normal2[3]) / 2
    possible_ls = find_possible_ls(normal1, normal2, angle_dict)
    possible_ls = [n2 for n1 in possible_ls for n2 in n1 if n2]
    possible_ls_test = [ls[0] for ls in possible_ls]
    if not possible_ls:
        print("error")
        return 0
    loss_ls = []
    for possible in possible_ls_test:
        loss, parm = find_loss_and_parameter(d1, d2, possible, real_data1, real_data2,normal1,normal2)
        loss_ls.append(loss)
    min_idx = 0
    for idx in range(1, len(loss_ls)):
        if loss_ls[idx] < loss_ls[min_idx]:
            min_idx = idx
    possible_ls_want = possible_ls[min_idx]
    return possible_ls_want


def find_loss_and_parameter(d1, d2, possible, real_data1, real_data2,normal1,normal2):
    global_x, loss = random_walk_functions.find_parameter(possible, normal1[:2], normal2[:2])

    rx = (possible[0] * PI) / 180
    ry = (possible[1] * PI) / 180
    rz = 0
    c1, c2, h1, dh = global_x

    _, mic_data1 = ring_functions.make_mimic_data_3d(rx, ry, rz, d1, h1, 0, 180, c1, c2)
    pin1, idx_ls1 = find_pin(mic_data1, real_data1)
    _, mic_data2 = ring_functions.make_mimic_data_3d(rx, ry, rz, d2, h1 + dh, 0, 180, c1, c2)
    pin2, idx_ls2 = find_pin(mic_data2, real_data2)

    crop_mic_data1 = mic_data1[2, idx_ls1]
    new_real_data1 = np.vstack([real_data1[0], real_data1[1], crop_mic_data1])
    rotated_back1_real = np.dot(np.linalg.inv(ring_functions.rotatexyz(rx, ry, rz)), new_real_data1)

    crop_mic_data2 = mic_data2[2, idx_ls2]
    new_real_data2 = np.vstack([real_data2[0], real_data2[1], crop_mic_data2])
    rotated_back2_real = np.dot(np.linalg.inv(ring_functions.rotatexyz(rx, ry, rz)), new_real_data2)

    loss = loss + np.std(rotated_back2_real[2]) + np.std(rotated_back1_real[2]) + ring_functions.find_std_angle(
        idx_ls1) + ring_functions.find_std_angle(idx_ls2)
    parm = [global_x, rotated_back1_real, rotated_back2_real, possible]
    return loss, parm


def find_parm_want(normal1, normal2, possible_ls_want, real_data1, real_data2):
    d1 = max(normal1[2], normal1[3]) / 2
    d2 = max(normal2[2], normal2[3]) / 2
    min_loss = np.inf
    for idx, possible in enumerate(possible_ls_want):
        loss, parm = find_loss_and_parameter(d1, d2, possible, real_data1, real_data2,normal1,normal2)
        if loss < min_loss:
            min_loss = loss
            global_parm = parm
    print(min_loss)
    return global_parm


def find_pin(mic_data1, real_data):
    """ nearest localization in two dimention

    """
    mic_data1_c = mic_data1[:2]
    idx_ls = []
    sum_loss = 0
    for idx in range(real_data.shape[1]):
        pos = real_data[:, idx]
        x, y = pos
        dx = mic_data1[0] - x
        dy = mic_data1[1] - y

        dv = np.vstack([dx, dy])
        loss_array = np.sum((dv * dv), axis=0)
        idx, loss_n = loss_functions.help_loss_function3_find_idx(loss_array, idx_ls)
        idx_ls.append(idx)
        sum_loss += loss_n
    # print(idx_ls)
    # dic_l=idx_ls
    # d_ls=[]
    # for n in range(1,9):
    # d_ls.append(max(dic_l[n],dic_l[n-1])-min(dic_l[n],dic_l[n-1]))

    # d_ls.append(dic_l[-1]-dic_l[0])

    # for idx,x in enumerate(d_ls):
    # if x > 90:
    # d_ls[idx]=180-x
    # sum(map(abs,[x-20 for x in d_ls]))
    return sum_loss, idx_ls


def re_rote_matrix(rotate_matrix,rotated_data,idx_ls,real_data,hext,ratio):
    global_ls=[]
    for i in range(9):
        global_x,loss=random_walk_functions.find_parameter_position(rotated_data[:,idx_ls[i]],real_data[:,i],rotate_matrix,hext,ratio)
        global_ls.append(global_x)
    return global_ls