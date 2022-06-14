import numpy as np
from PIL import Image
import os
# 28*28 이미지 변환한 것을 numpy 배열로 바꾸는 코드 
# 이미지 파일 경로 D:\image

def image_to_numpy_array():
    path_dir = 'D:\\resize_image\\'
    Destination_path = 'C:\\Users\\SCHCsRC\\Desktop\\병열\\Python\Machine-Learning-Classification\\numpy_array\\'
    file_list = os.listdir(path_dir)

    # np_list=[]
    # for png in file_list:
    #     image = Image.open(path_dir + png)
    #     pixel = np.array(image)
    #     png = png.split('.')[0]
    # np.savez_compressed(Destination_path+'data', png)
    pixel = np.array(file_list) # Numpy 배열 생성
    print(pixel)
    np.savez_compressed(Destination_path+'data', pixel) # 생성된 배열을 1개의 압축되지 않은 .npz 파일로 저장