import  numpy as np

def read_txt(file_path):
    f=open(file_path)
    f.readline()
    xl=[]
    yl=[]
    for n in range(9):
        lx=f.readline()
        lx=lx.rstrip()
        lx=lx.split("\t")
        xl.append(float(lx[0]))
        yl.append(float(lx[1]))
        #xl.append(float(lx[0])
        #yl.append(float(lx[1])
    return np.array([xl,yl])