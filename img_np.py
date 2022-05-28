import numpy as np
from PIL import Image
import os

# 이미지 파일 경로 D:\image
path_dir = ''# '' 안에 파일을 묶고 있는 "폴더 경로"를 쓰세요
file_list = os.listdir(path_dir)

for png in file_list:
    image = Image.open(path_dir + png)
    pixel = np.array(image)
    png = png.split('.')[0]
    np.save(""+png, pixel) #저장할 '폴더 경로'를 쓰세요

