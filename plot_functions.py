

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

red =ImageColor.getrgb("red")
magenta =ImageColor.getrgb("magenta")
blue= ImageColor.getrgb("blue")
green= ImageColor.getrgb("green")
cyan= ImageColor.getrgb("cyan")
black= ImageColor.getrgb("black")
white= ImageColor.getrgb("white")
colordict ={"red" :red ,"magenta" :magenta ,"green" :green ,"cyan" :cyan ,"blue" :blue ,"black" :black,"white":white}


def plot2ring_and_star(fitdata1 ,fitdata2 ,real_data1 ,real_data2 ,color_dict=colordict,name="saved_demo"):
    name="saved_image/"+name+".png"
    sp =int(max(max(real_data1[0]) ,max(real_data1[1]) ,max(real_data2[0]) ,max(real_data2[1])) ) +20
    image =np.ones([sp ,sp ,3])


    for idx in range(real_data1.shape[1]):
        pos =real_data1[: ,idx]

        for dpx in [-1,0,1]:
            for dpy in [-1,0,1]:
                image[int(pos[1]+dpy) ,int(pos[0]+dpx) ,: ] =color_dict["green"]


    for idx in range(real_data2.shape[1]):
        pos =real_data2[: ,idx]
        for dpx in [-1,0,1]:
            for dpy in [-1,0,1]:
                image[int(pos[1]+dpy) ,int(pos[0]+dpx) ,: ] =color_dict["magenta"]



    for idx in range(len(fitdata1[0])):
        pos =(fitdata1[0][idx] ,fitdata1[1][idx])
        image[int(pos[1]) ,int(pos[0]) ,: ] =color_dict["white"]
        image[int(pos[1 ] +1) ,int(pos[0]) ,: ] =color_dict["white"]
        image[int(pos[1 ] -1) ,int(pos[0]) ,: ] =color_dict["white"]
        image[int(pos[1]) ,int(pos[0 ] +1) ,: ] =color_dict["white"]
        image[int(pos[1]) ,int(pos[0 ] -1) ,: ] =color_dict["white"]

    for idx in range(len(fitdata2[0])):
        pos =(fitdata2[0][idx] ,fitdata2[1][idx])
        image[int(pos[1]) ,int(pos[0]) ,: ] =color_dict["white"]
        image[int(pos[1 ] +1) ,int(pos[0]) ,: ] =color_dict["white"]
        image[int(pos[1 ] -1) ,int(pos[0]) ,: ] =color_dict["white"]
        image[int(pos[1]) ,int(pos[0 ] +1) ,: ] =color_dict["white"]
        image[int(pos[1]) ,int(pos[0 ] -1) ,: ] =color_dict["white"]



    new_im = Image.fromarray(np.uint8(image))
    new_im.save(name)





def plot2ring3D(data1, data2,ratio=0, elevn=40,filename= "saved_demo"):

    filename="saved_image/"+str(ratio)+"_"+str(elevn)+filename+".pdf"
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d', elev=elevn)
    x1, y1, z1 = data1[0, :], data1[1, :], data1[2, :]
    x2, y2, z2 = data2[0, :], data2[1, :], data2[2, :]

    x1 = np.hstack([x1, x1[0]])
    y1 = np.hstack([y1, y1[0]])
    z1 = np.hstack([z1, z1[0]])

    x2 = np.hstack([x2, x2[0]])
    y2 = np.hstack([y2, y2[0]])
    z2 = np.hstack([z2, z2[0]])

    ax.plot(x1, y1, z1, color="g", marker="o")
    ax.plot(x2, y2, z2, color="m", marker="o")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_yticks([])
    temp = np.sum(data1[2, :]) / len(data1[2])
    zlim = [temp - 20, temp + 20]
    #     ax2 = fig.add_subplot(212)
    #     ax2.plot(data1[0,:],data1[1,:],color="b",marker="o")
    #     ax2.plot(data2[0,:],data2[1,:],color="r",marker="o")
    ax.set_zlim(zlim[0], zlim[1])
    plt.tight_layout()
    plt.savefig(filename)


