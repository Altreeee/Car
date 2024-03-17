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



import TestForAutoD
import pd_final_3.test


#script,vels,save_path= argv

def dataset(video): 
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

if __name__ == "__main__":
    cout = 0
    # 从当前目录中读取 "test.jpg" 图片
    image_path = "stop.jpg"
    # 读取图像
    #image = cv2.imread(image_path)

    vels  = 1545
    #print("reach line 67")

    try:
        while 1:
            vel = int(vels)
            '''img = dataset(image)'''
            print("reach line 72")

            direction, rate = TestForAutoD.process_image(image_path)
            print(direction)
            print(rate)
            print("line 74 in main programme")

            a = 1500
            if abs(rate)>10 and abs(rate)<50 and direction == "left":
                '''需要右转'''
                a = 1350
            if abs(rate)>50 and direction == "left":
                '''急需右转'''
                a = 1200
            if abs(rate)>10 and abs(rate)<50 and direction == "right":
                '''需要左转'''
                a = 1650
            if abs(rate)>50 and direction == "right":
                '''急需左转'''
                a = 1800

            
            send_data = pd_final_3.test.infer(image_path)
            print("reach get send_data")
            if send_data != 'null':
                vel = 0
            print("end of if send_data")

            #execfile('//home//deep//paddlepaddle//pd_6//test.py')
            '''lib.send_cmd(vel, a)'''
            print(cout)
            cout=cout+1
            print("angle: %d, throttle: %d" % (a, vel))

    except:
        print('error')
    finally:
        '''lib.send_cmd(1500, 1500)'''


