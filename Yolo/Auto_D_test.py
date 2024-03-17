# -*- coding:utf-8 -*-
"""

run image---->angle

get data from camera:/dev/video0   

params:vels=1535

put image to serial :/dev/ttACM0


"""


import select
from ctypes import *
import numpy as np
import cv2 
from sys import argv


#import paddle.fluid as fluid
from PIL import Image




import test


#script,vels,save_path= argv

'''def dataset(video): 
    lower_hsv = np.array([19,20,210])
    upper_hsv = np.array([40, 255, 255])

    select.select((video,), (), ())        
    image_data = video.read_and_queue()
    
    frame = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask0 = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)#在上下限范围内的为白色，不符合的为黑色
    #mask1 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=upper_hsv1)
    mask = mask0 #+ mask1


    img = Image.fromarray(mask)
    img = img.resize((120, 120), Image.ANTIALIAS)
    img = np.array(img).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img.transpose((2, 0, 1))
    img = img[(2, 1, 0), :, :] / 255.0
    img = np.expand_dims(img, axis=0)   
    return img
'''

def dataset(video):
    select.select((video,), (), ())        
    image_data = video.read_and_queue()
    # 将图像数据解码为OpenCV格式
    frame = cv2.imdecode(
        np.frombuffer(image_data, dtype=np.uint8), 
        cv2.IMREAD_COLOR
    )
    '''cv2.imwrite(f'frame_{frame_number}.jpg', frame)'''
    return frame



if __name__ == "__main__":
    cout = 0
    cam = cv2.VideoCapture(0)   #打开编号为00的摄像头【0号是上面的摄像头】

    vels  = 1545
    '''lib_path = path + "/lib" + "/libart_driver.so"'''
    lib_path = "./lib/libart_driver.so"
    so = cdll.LoadLibrary
    lib = so(lib_path)
    car = "/dev/ttyACM0"
    #print("reach line 67")
    
    if (lib.art_racecar_init(38400, car.encode("utf-8")) < 0):
        raise
        pass
    try:
        while 1:
            vel = int(vels)
            ret,frame = cam.read()
            res=cv2.resize(frame,(224,224),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite("1.jpg",res)
            image_path = "1.jpg"
		    
            print("reach line 72")

            a = 1500
            n_labels = test.infer(image_path)
            print("reach get send_data")
            if 2 in n_labels:
                print("找到了名为'停止'的标签！")
                vel = 1500
            if 0 in n_labels:
                print("找到了名为'人'的标签！")
                vel = 1500
            if 1 in n_labels:
                print("找到了名为'斑马线'的标签！")
                vel = 1500
            if 3 in n_labels:
                print("找到了名为'左'的标签！")
                a = 1650
            if 5 in n_labels:
                print("找到了名为'右'的标签！")
                a = 1290

            print("end of if send_data")

            #execfile('//home//deep//paddlepaddle//pd_6//test.py')
            lib.send_cmd(vel, a)
            print(cout)
            cout=cout+1
            print("angle: %d, throttle: %d" % (a, vel))

    except:
        print('error')
    finally:
        lib.send_cmd(1500, 1500)


