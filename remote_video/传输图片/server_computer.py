#!/usr/bin/env python
# -*- coding=utf-8 -*-
 
import socket
import numpy as np
import urllib
import cv2
import threading
import time
import sys
 
 
def socket_client():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('192.168.43.145', 8090))#连接服务端
    except socket.error as msg:
        print (msg)
        sys.exit(1)
    
    print('this is Client')
    while True:
        receive_encode = s.recv(2147483647)#接收的字节数 最大值 2147483647 （31位的二进制）
        nparr = np.fromstring(receive_encode, dtype='uint8')
        img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imshow("img_decode", img_decode)#显示图片
        cv2.waitKey(1)
 
socket_client()