#!/usr/bin/env python
# -*- coding=utf-8 -*-
# https://blog.csdn.net/weixin_54835709/article/details/122452388?spm=1001.2014.3001.5502
# https://blog.csdn.net/weixin_54835709/article/details/122452388?spm=1001.2014.3001.5502
 
import socket
import numpy as np
import urllib
import cv2 as cv
import threading
import time
import sys
 
 
print('this is Server')
 
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 260)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 200)
 
def socket_service():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 防止socket server重启后端口被占用（socket.error: [Errno 98] Address already in use）
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #s.bind(('127.0.0.1', 6666))
        s.bind(('', 8090))#这个是服务端机器的ip
        s.listen(True)
    except socket.error as msg:
        print (msg)
        sys.exit(1)
    print ('Waiting connection...')
 
    while True:
        conn, addr = s.accept()
        t = threading.Thread(target=deal_data, args=(conn, addr))
        t.start()
 
def deal_data(conn, addr):
    #print ('Accept new connection from {0}'.format(addr))
    while True:
        # get a frame
        ret, frame = cap.read()
        # '.jpg'表示把当前图片img按照jpg格式编码，按照不同格式编码的结果不一样
        img_encode = cv.imencode('.jpg', frame)[1] 
        data_encode = np.array(img_encode)
        str_encode = data_encode.tostring()
        encode_len = str(len(str_encode))
        print('img size : %s'%encode_len)
        try:
            conn.send(str_encode)#发送图片的encode码
        except Exception as e:
            print(e)
        time.sleep(0.1)
    conn.close()
 
socket_service()
 
