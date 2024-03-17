'''
按类别移动图片和xml文件的位置
'''

import os
import shutil
import xml.etree.ElementTree as ET

def move_files(xml_folder):
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_folder, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name == 'Straight':
                    filename = root.find('filename').text
                    data_path = os.path.join(os.path.dirname(xml_folder), 'jpg')
                    jpg_path = os.path.join(data_path, filename)
                    destination_folder_xml = os.path.join(os.path.dirname(xml_folder), 'folder1')
                    destination_folder_jpg = os.path.join(os.path.dirname(xml_folder), 'folder2')
                    if not os.path.exists(destination_folder_xml):
                        os.makedirs(destination_folder_xml)
                    if not os.path.exists(destination_folder_jpg):
                        os.makedirs(destination_folder_jpg)
                    shutil.move(xml_path, os.path.join(destination_folder_xml, xml_file))
                    shutil.move(jpg_path, os.path.join(destination_folder_jpg, filename))

# 使用示例
xml_folder = 'xml'
move_files(xml_folder)
