'''
让小车不断拍照
'''


import os
import cv2

if __name__ == '__main__':
    cout = 0
    cam = cv2.VideoCapture(0)   # 打开编号为0的摄像头【0号是上面的摄像头】
    while 1:
        ret, frame = cam.read()
        res = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
        folder_path = "pic"  # 图片保存文件夹路径
        if not os.path.exists(folder_path):  # 如果文件夹不存在，则创建
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, "{}.jpg".format(cout))  # 图片文件路径
        cv2.imwrite(file_path, res)  # 保存图片
        cout += 1  # 每次循环结束，cout加1
        
        # 检测键盘输入
        key = cv2.waitKey(1)
        if key == ord('i'):  # 如果按下 'i' 键
            break

        print(cout)

    cam.release()
    cv2.destroyAllWindows()
