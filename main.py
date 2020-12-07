print("this work")

def data_and_normal(file_path):
    real_data=IO_functions.read_txt(file_path)
    paras=ellipse_fitting.solve_tuoyuan(real_data[0],real_data[1])
    normal=ellipse_fitting.normal_style(paras)
    return real_data,normal




import utils
import numpy as np
import IO_functions
import ellipse_fitting
import find_parameter
import ring_functions
import plot_functions
import time
st=time.time()
PI= np.pi

#angle_dict=make_ratio_and_angle()
#np.save('angle_dict.npy', angle_dict)




utils.make_sure_imported()

read_dictionary = np.load('angle_dict.npy',allow_pickle='TRUE').item() ## 180 * 180

import info_dict


def main(file_path1, file_path2, read_dictionary):
    real_data1, normal1 = data_and_normal(file_path1)
    real_data2, normal2 = data_and_normal(file_path2)
    possible_ls_want = find_parameter.find_which_possible(normal1, normal2, read_dictionary, real_data1, real_data2)
    possible_ls_want = utils.fitness_possible_ls(possible_ls_want)
    global_parm = find_parameter.find_parm_want(normal1, normal2, possible_ls_want, real_data1, real_data2)

    parms = utils.reformat_parm(normal1, normal2, global_parm)  # rx,ry,rz,c1,c2,h1,dh,d1,d2=parms
    rx, ry, rz, c1, c2, h1, dh, d1, d2 = parms

    rotate_matrix, data1, data2 = ring_functions.reconstruct_data(parms)

    global_d_want, global_d1xy_loss = ring_functions.help_find_mi_rotation(data1, data2, rotate_matrix, real_data1,
                                                                           real_data2)

    #print(global_d1xy_loss)
    dx, dy, rotated_data1, rotated_data2, roted_mic_data1, roted_mic_data2, idx_ls1, idx_ls2 = global_d_want
    print(dx, dy)

    rotated_back1_real = ring_functions.reverse_rotation(idx_ls1, roted_mic_data1, real_data1, parms)
    rotated_back2_real = ring_functions.reverse_rotation(idx_ls2, roted_mic_data2, real_data2, parms)
    mean_h1 = np.mean(rotated_back1_real[2, :])
    mean_h2 = np.mean(rotated_back2_real[2, :])

    ratio = 0.2
    new_localization_ls1 = find_parameter.re_rote_matrix(rotate_matrix, rotated_data1, idx_ls1, real_data1, mean_h1,
                                                         ratio=ratio)
    new_localization_ls2 = find_parameter.re_rote_matrix(rotate_matrix, rotated_data2, idx_ls2, real_data2, mean_h2,
                                                         ratio=ratio)

    new_localization1 = np.vstack(new_localization_ls1).T
    new_localization2 = np.vstack(new_localization_ls2).T
    maxdz1=max(new_localization1[2])-min(new_localization1[2])
    maxdz2 = max(new_localization2[2]) - min(new_localization2[2])

    theta = ring_functions.find_theta(idx_ls1, idx_ls2)


    file=open("data_info/"+filename+".txt","w")
    file.write("d1:{} ,d2:{} ,dh:{} ,theta:{} ,max_dz1:{},max_dz2:{}".format(d1, d2, dh, theta,maxdz1,maxdz2))
    end = time.time()
    print("time consumed :{}".format(end - st))
    plot_functions.plot2ring3D(new_localization1, new_localization2, ratio, elevn=0, filename=filename)
    plot_functions.plot2ring3D(new_localization1, new_localization2, ratio, elevn=90, filename=filename)
    plot_functions.plot2ring3D(new_localization1, new_localization2, ratio, elevn=40, filename=filename)



    mic_data1 = np.dot(rotate_matrix, new_localization1)
    mic_data2 = np.dot(rotate_matrix, new_localization2 )
    plot_functions.plot2ring_and_star(mic_data1[:2], mic_data2[:2], real_data1, real_data2, name=filename)





for k,v in info_dict.info_dict.items():
    filename=k
    file_path1=v[0]
    file_path2=v[1]
    main(file_path1,file_path2,read_dictionary)


