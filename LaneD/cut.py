import cv2

def extract_frame(video_path, frame_num, save_path):
    # 打开视频文件
    cap = cv2.VideoCapture("source.mp4")

    # 确保视频文件打开成功
    if not cap.isOpened():
        print("Error: 无法打开视频文件")
        return

    # 设置要提取的帧的位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    # 读取帧
    ret, frame = cap.read()

    # 如果成功读取帧，则保存图片
    if ret:
        cv2.imwrite(save_path, frame)
        print("图片已保存为:", save_path)
    else:
        print("Error: 无法读取指定帧")

    # 释放视频对象
    cap.release()

# 指定视频文件路径
video_file = 'your_video.mp4'

# 调用函数提取第2秒的帧并保存
extract_frame(video_file, 2 * 30, '1.jpg')  # 乘以30是因为视频默认的帧率是每秒30帧
