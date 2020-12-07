import random
import math
import numpy as np
from PIL import Image
from PIL import ImageColor
import utils
import loss_functions
def find_parameter(rxy, center1, center2):
    min_loss = np.inf
    round_num = 0
    while (round_num < 1):
        round_num += 1

        N = 100  # 迭代次数
        step = 10  # 初始步长
        epsilon = 0.05
        variables = 4  # 变量数目
        c1, c2 = center1[0], center1[1]
        h1, dh = [0, 0]
        x = [c1, c2, h1, dh]  # 初始点坐标 c1 c2 h1 dh
        walk_num = 1  # 初始化随机游走次数
        n = 10  # 每次随机生成向量u的数目
        #         print("round_number: ",round_num)
        #         print("迭代次数:",N)
        #         print("初始步长:",step)
        #         print("每次产生随机向量数目:",n)
        #         print("epsilon:",epsilon)
        #         print("变量数目:",variables)
        #         print("初始点坐标:",x)
        # 定义目标函数
        while (step > epsilon):
            k = 1  # 初始化计数器
            while (k < N):
                # 产生n个向量u
                x1_list = []  # 存放x1的列表
                for i in range(n):
                    u = [random.uniform(-1, 1) for i1 in range(variables)]  # 随机向量
                    # u1 为标准化之后的随机向量

                    u1 = []
                    for i3 in range(variables):
                        temp = 0
                        for i2 in range(variables):
                            temp += u[i2] ** 2
                        u1.append(u[i3] / math.sqrt(temp))

                    # u1 = [u[i3]/math.sqrt(sum([u[i2]**2 for i2 in range(variables)])) for i3 in range(variables)]
                    x1 = utils.update_variable(x, u1, step)
                    x1_list.append(x1)
                f1_list = []
                for parameter in x1_list:
                    # print(parameter)
                    loss = loss_functions.find_loss(center1, center2, parameter, rxy)
                    f1_list.append(loss)
                f1_min = min(f1_list)
                f1_index = f1_list.index(f1_min)
                x11 = x1_list[f1_index]  # 最小f1对应的x1
                if (f1_min < loss_functions.find_loss(center1, center2, x, rxy)):  # 如果找到了更优点
                    # print("min Loss :{}".format(min_loss))
                    k = 1
                    x = x11
                    # print(f1_min)
                    if f1_min < min_loss:
                        global_x = x11
                        min_loss = f1_min
                else:
                    k += 1
            step = step * 0.5

            # print("第%d次随机游走完成。" % walk_num)
            walk_num += 1
        # d1,d2=global_x[0],global_x[1]
    #         print("min Loss :{}".format(min_loss))
    #         print("parameter")
    #         print(global_x)
    #         print()

    # print("随机游走次数:",walk_num-1)
    # print("最终最优点:", global_x)
    # print("最终最优值:", min_loss)
    return global_x, min_loss


def find_parameter_position(circle_position, real_position,rotate_matrix, hext,ratio):


    min_loss = np.inf
    round_num = 0
    while (round_num < 1):
        round_num += 1
        N = 100  # 迭代次数
        step = 1  # 初始步长
        epsilon = 0.05
        variables = 3  # 变量数目
        x = [circle_position[0], circle_position[1], hext]  # 初始点坐标 xyz
        walk_num = 1  # 初始化随机游走次数
        n = 8  # 每次随机生成向量u的数目
        while (step > epsilon):
            k = 1  # 初始化计数器
            while (k < N):
                x1_list = []  # 存放x1的列表
                for i in range(n):
                    u = [random.uniform(-1, 1) for i1 in range(variables)]  # 随机向量
                    u1 = []
                    for i3 in range(variables):
                        temp = 0
                        for i2 in range(variables):
                            temp += u[i2] ** 2
                        u1.append(u[i3] / math.sqrt(temp))
                    x1 = utils.update_variable_z(x, u1, step,ratio)
                    x1_list.append(x1)
                f1_list = []
                for parameter in x1_list:
                    loss = loss_functions.find_loss_position(parameter, rotate_matrix,real_position)
                    f1_list.append(loss)
                f1_min = min(f1_list)
                f1_index = f1_list.index(f1_min)
                x11 = x1_list[f1_index]  # 最小f1对应的x1
                if (f1_min < loss_functions.find_loss_position(x, rotate_matrix,real_position)):  # 如果找到了更优点
                    k = 1
                    x = x11
                    if f1_min < min_loss:
                        global_x = x11
                        min_loss = f1_min
                else:
                    k += 1
            step = step * 0.5
            walk_num += 1
    # print("最终最优点:", global_x)
    # print("最终最优值:", min_loss)
    return global_x, min_loss