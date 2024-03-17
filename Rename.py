'''
给文件夹中的文件重命名，按顺序从0开始
'''

import os

def rename_photos(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    # 筛选出照片文件
    photo_files = [file for file in files if file.lower().endswith(('.xml'))]

    # 遍历并重命名照片文件
    for i, photo_file in enumerate(photo_files, start=0):
        old_path = os.path.join(folder_path, photo_file)
        new_path = os.path.join(folder_path, f"{i}.xml")

        # 如果目标文件已存在，则尝试加一后重命名
        count = 1
        while os.path.exists(new_path):
            new_path = os.path.join(folder_path, f"{i + count}.xml")
            count += 1

        os.rename(old_path, new_path)

if __name__ == "__main__":
    folder_path = input("请输入文件夹路径：")
    rename_photos(folder_path)
    print("照片重命名完成！")
