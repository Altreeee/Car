#!/usr/bin/env python
# -*- coding=utf-8 -*-
 
import socket
import numpy as np
import urllib
import cv2
import threading
import sys
import time
 
 
print('this is Server')
 
img = cv2.imread('1.jpg')#这个是我本地的图片，和这个py文件在同一文件夹下，注意格式
# '.jpg'表示把当前图片img按照jpg格式编码，按照不同格式编码的结果不一样
img_encode = cv2.imencode('.jpg', img)[1]
 
data_encode = np.array(img_encode)
str_encode = data_encode.tostring()
encode_len = str(len(str_encode))
print(encode_len)#输出看一下encode码的大小，可有可无
 
def socket_service():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('', 8090))
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
    #print ('Accept new connection from {0}').format(addr)
    while True:
        conn.send(str_encode)#发送图片的encode码
        time.sleep(1)
    conn.close()
 
socket_service()