#!/usr/bin/python
# -*-coding:utf-8 -*-
import socket
import cv2
import numpy
from time import sleep

# socket.AF_INET 用于服务器与服务器之间的网络通信
# socket.SOCK_STREAM 代表基于TCP的流式socket通信
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 连接服务端
address_server = ('10.0.0.30', 8888)
sock.connect(address_server)

# 从摄像头采集图像
# 参数是0,表示打开笔记本的内置摄像头,参数是视频文件路径则打开视频
'''capture = cv2.VideoCapture(0)'''
capture = "source.jpg"
frame = cv2.imread(capture)

if frame is not None:
    
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]

    # 首先对图片进行编码，因为socket不支持直接发送图片
    # '.jpg'表示把当前图片frame按照jpg格式编码
    # result, img_encode = cv2.imencode('.jpg', frame)
    img_encode = cv2.imencode('.jpg', frame, encode_param)[1]
    data = numpy.array(img_encode)
    stringData = data.tostring()
    # 首先发送图片编码后的长度
    sock.send(str.encode(str(len(stringData)).ljust(16)))
    # 然后一个字节一个字节发送编码的内容
    # 如果是python对python那么可以一次性发送，如果发给c++的server则必须分开发因为编码里面有字符串结束标志位，c++会截断
    # for i in range(0, len(stringData)):
    #     sock.send(stringData[i])
    sock.send(stringData)
    # sleep(1)
    ret, frame = capture.read()
    cv2.resize(frame, (640, 480))

sock.close()
cv2.destroyAllWindows()
