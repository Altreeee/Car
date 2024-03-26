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
    #c_sock, c_addr =s.accept()
    print('this is Client')
    while True:
        try:
            receive_encode = s.recv(2147483647)#接收的字节数 最大值 2147483647 （31位的二进制）
            nparr = np.fromstring(receive_encode, dtype='uint8')
            img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imshow("Client_show", img_decode)
            cv2.waitKey(1)
        except Exception as e:
            print(e)
        
 
socket_client()
 