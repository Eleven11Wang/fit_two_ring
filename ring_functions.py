import numpy as np
import loss_functions
PI=np.pi
def rotatexyz(x, y, z):
    # xyx rotate angle
    rotatex = [[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]]
    rotatey = [[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]]
    rotatez = [[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]]
    temp = np.dot(rotatez, rotatey)
    return np.dot(temp, rotatex)


def make_mimic_data_3d(rx, ry, rz, d1, h1, theta, n, c1, c2):
    rotation_matrix = rotatexyz(rx, ry, rz)
    u = np.linspace(0, 2 * np.pi, n, endpoint=False) + theta * PI

    x1 = d1 * (np.sin(u)) + c1
    y1 = d1 * (np.cos(u)) + c2
    z1 = h1 * np.ones(x1.shape)
    data_stack = np.vstack([x1, y1, z1])
    mic_data = np.dot(rotation_matrix, data_stack)
    return [data_stack, mic_data]


def find_std_angle(idx_ls1):
    d_idx_1 = []
    idx_ls1 = sorted(idx_ls1)
    for i in range(1, len(idx_ls1)):
        d_idx_1.append(idx_ls1[i] - idx_ls1[i - 1])
    d_idx_1 = [x - 40 for x in d_idx_1]
    return np.std(d_idx_1)


def ratio_ring(normal):
    return max(normal[3], normal[2]) / min(normal[3], normal[2])


def reverse_rotation(idx_ls,roted_mic_data,real_data,parms):
    rx, ry, rz, c1, c2, h1, dh, d1, d2 = parms

    crop_mic_data=roted_mic_data[2,idx_ls]
    new_real_data=np.vstack([real_data[0],real_data[1],crop_mic_data])
    rotated_back_real=np.dot(np.linalg.inv(rotatexyz(rx,ry,rz)),new_real_data)
    return rotated_back_real


def find_theta(idx_ls1,idx_ls2):
    idx_ls1=sorted(idx_ls1)
    idx_ls2=sorted(idx_ls2)
    sumx=0
    for idx in range(len(idx_ls1)):
        sumx+=idx_ls1[idx]-idx_ls2[idx]
    return 2*sumx/9


def help_find_mi_rotation(data1, data2, rotate_matrix, real_data1, real_data2):
    global_d1xy_loss = np.inf
    for dx in np.linspace(-5, 5, 10, endpoint=False):
        for dy in np.linspace(-5, 5, 10, endpoint=False):
            rdx = dx * PI / 180
            rdy = dy * PI / 180

            rotate_matrix_rst = rotatexyz(rdx, rdy, 0)
            rotated_data1 = np.dot(rotate_matrix_rst, data1)
            rotated_data2 = np.dot(rotate_matrix_rst, data2)

            rotated_mic_data1 = np.dot(rotate_matrix, rotated_data1)
            rotated_mic_data2 = np.dot(rotate_matrix, rotated_data2)

            idx_ls1, loss1 = matrix_loss(rotated_mic_data1, real_data1)
            idx_ls2, loss2 = matrix_loss(rotated_mic_data2, real_data2)

            loss = loss1 + loss2

            if loss < global_d1xy_loss:
                global_d_want = (
                dx, dy, rotated_data1, rotated_data2, rotated_mic_data1, rotated_mic_data2, idx_ls1, idx_ls2)
                global_d1xy_loss = loss
    return global_d_want, global_d1xy_loss

def reconstruct_data(parms):
    rx, ry, rz, c1, c2, h1, dh, d1, d2 = parms
    rotate_matrix = rotatexyz(rx, ry, rz)
    data1, mic_data1_b = make_mimic_data_3d(rx, ry, rz, d1, h1, 0, 180, c1, c2)
    data2, mic_data2_b = make_mimic_data_3d(rx, ry, rz, d2, h1 + dh, 0, 180, c1, c2)
    return rotate_matrix,data1,data2




def help_find_mi_rotation(data1,data2,rotate_matrix,real_data1,real_data2):
    global_d1xy_loss=np.inf
    for dx in np.linspace(-5,5,10,endpoint=False):
        for dy in np.linspace(-5,5,10,endpoint=False):
            rdx=dx*PI/180
            rdy=dy*PI/180

            rotate_matrix_rst=rotatexyz(rdx,rdy,0)
            rotated_data1=np.dot(rotate_matrix_rst,data1)
            rotated_data2=np.dot(rotate_matrix_rst,data2)

            rotated_mic_data1=np.dot(rotate_matrix,rotated_data1)
            rotated_mic_data2=np.dot(rotate_matrix,rotated_data2)

            idx_ls1,loss1=loss_functions.matrix_loss(rotated_mic_data1,real_data1)
            idx_ls2,loss2=loss_functions.matrix_loss(rotated_mic_data2,real_data2)

            loss= loss1+ loss2

            if loss < global_d1xy_loss:
                global_d_want=(dx,dy,rotated_data1,rotated_data2,rotated_mic_data1,rotated_mic_data2,idx_ls1,idx_ls2)
                global_d1xy_loss=loss
    return global_d_want,global_d1xy_loss


