import cv2
import random
from PIL import Image
import numpy as np


def getPic(pic1, pic2, dir):
    pic1_shape = np.array(pic1.shape)
    pic2_shape = np.array(pic2.shape)

    pic1_center = pic1_shape / 2-1
    pic2_center = pic2_shape / 2
    pic2_center = np.ceil(pic2_center)
    pic2_center = np.array(pic2_center,dtype=np.int16)

    rec_btn = pic1_shape[0] - (pic2_center[0])
    rec_rig = pic1_shape[1] - (pic2_center[1])

    rect = [pic2_center[0], rec_btn, pic2_center[1], rec_rig]
    hight = int((rect[1] - rect[0] )/ 2)
    length = int((rect[3] - rect[2] )/ 2)
    y = random.randint(0, int(length)-5)
    x = random.randint(0, int(hight)-5)

    if dir % 2 == 1:
        if dir == 1:
            new_x = pic1_center[0] - x
            new_y = pic1_center[1] + y
        else:
            new_x = pic1_center[0] + x
            new_y = pic1_center[1] - y
    else:
        if dir == 2:
            new_x = pic1_center[0] - x
            new_y = pic1_center[1] - y
        else:
            new_x = pic1_center[0] + x
            new_y = pic1_center[1] + y

    loc = [new_x - pic2_center[0], new_x + pic2_center[0], new_y - pic2_center[1], new_y + pic2_center[1]]
    loc = np.array(loc, dtype=np.int16)
    print(loc[0],loc[0]+pic2_shape[0]-1,loc[2],loc[2]+pic2_shape[1]-1)
    print(pic2_shape)

    pic1[loc[0]:loc[0]+pic2_shape[0], loc[2]:loc[2]+pic2_shape[1]] += pic2

    return pic1

def getaMoS(pic):
    img = Image.fromarray(pic)
    angle = random.randint(0,361)
    img = img.rotate(angle,expand=True)
    rate = random.uniform(0.5,1.4)
    img = np.array(img)
    return cv2.resize(img,None,fx=rate,fy=rate,interpolation=cv2.INTER_CUBIC)


backg = np.load('./origin/background.npy')
star  = np.load('./origin/star_only.npy')
moon = np.load('./origin/moon_only.npy')

for i in range(1000):
    newOpic = getaMoS(star)
    if i < 500:
        dic = 1
    else:
        dic = 3
    backg = np.load('./origin/background.npy')
    result = getPic(backg,newOpic,dic)
    np.save('./testset/star/s1_{0}.npy'.format(i),result)
