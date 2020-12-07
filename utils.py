

import numpy as np
PI=np.pi
import ring_functions

def make_ratio_and_angle():
    answ_dict = {}

    for idxx in range(180):
        for idxy in range(180):
            zr = 0
            data, mic_data_3d = ring_functions.make_mimic_data_3d((idxx * PI) / 180, (idxy * PI) / 180, 0, 1, 0, 0, 180, 0, 0)
            mic_data = mic_data_3d[:2]
            max_idx = 0
            min_idx = 0
            max_num = -np.inf
            min_nim = np.inf
            for i in range(0, mic_data.shape[1] // 2):
                d_data = mic_data[:, i] - mic_data[:, i + 90]
                dx = np.sqrt(sum(d_data * d_data))
                if dx > max_num:
                    max_idx = i
                    max_num = dx
                elif dx < min_nim:
                    min_idx = i
                    min_nim = dx
            r_min = mic_data[1][min_idx] / (mic_data[0][min_idx])
            r_max = mic_data[1][max_idx] / (mic_data[0][max_idx])
            answ_want = [round(max_num, 2), round(min_nim, 2), round(r_max, 2), round(r_min, 2)]
            answ_dict[(idxx, idxy)] = answ_want
    return answ_dict



def update_variable(x,ul,step):
    new_x_ls=[]
    for idx in range(len(ul)):
        new_x_ls.append(x[idx]+step*ul[idx])
    return new_x_ls

def update_variable_z(x,ul,step,ratio):
    new_x_ls=[]
    for idx in range(len(ul)-1):
        new_x_ls.append(x[idx]+step*ul[idx])
    new_x_ls.append(x[-1]+ratio*step*ul[-1])
    return new_x_ls

def find_center(mic_data):
    max_idx = 0
    min_idx = 0
    max_num = -np.inf
    min_nim = np.inf
    for i in range(0, mic_data.shape[1] // 2):
        d_data = mic_data[:, i] - mic_data[:, i + 90]
        dx = np.sqrt(sum(d_data * d_data))
        if dx > max_num:
            max_idx = i
            max_num = dx
        elif dx < min_nim:
            min_idx = i
            min_nim = dx
    center1 = (mic_data[:, max_idx] + mic_data[:, max_idx + 90]) / 2
    # center2=(mic_data[:,min_idx]+mic_data[:,min_idx+90])/2
    # print(center1)


def fitness_possible_ls(possible_ls_want):
    angle_dict = {}
    for angle in possible_ls_want:
        anglex, angley = angle
        if anglex not in angle_dict:
            angle_dict[anglex] = [angley]
        else:
            inset_y = angle_dict[anglex]
            temp = [y + angley for y in inset_y]
            if 180 not in temp:
                angle_dict[anglex].append(angley)
    return_ls = []
    for k, v in angle_dict.items():
        for y in v:
            return_ls.append((k, y))

    return return_ls


def reformat_parm(normal1,normal2,global_parm):
    global_x, rotated_back1_real, rotated_back2_real, possible = global_parm
    rx=(possible[0] *PI)/180
    ry=(possible[1] *PI)/180
    rz=0
    c1,c2,h1,dh=global_x
    d1=max(normal1[2],normal1[3])/2
    d2=max(normal2[2],normal2[3])/2
    re_ls=[rx,ry,rz,c1,c2,h1,dh,d1,d2]
    return re_ls

def make_sure_imported():
    print("utils imported")

