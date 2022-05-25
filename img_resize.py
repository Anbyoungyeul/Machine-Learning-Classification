import image_change
from PIL import Image
import cv2
import numpy as np
import os

def image_resize():
    image_path = 'D:\\image\\' # 원래 이미지 경로
    resized_path = 'D:\\resize_image\\' # 이미지 수정 경로 

    file_list = os.listdir(image_path)
    img_list=[]

    for image_list in file_list:
        if image_list.find('.png'): # 확장자 검색
            img_list.append(image_list) # 각 png 파일 리스트에 추가

    total_image = len(img_list)
    index = 0 # 구분 인덱스 
    for name in img_list:
        img=Image.open('%s%s' %(image_path, name))
        img_array = np.array(img) # 배열로 다시
        img_resize = cv2.resize(img_array, (28,28), interpolation=cv2.INTER_AREA) #255x255로 수장w정
        img=Image.fromarray(img_resize) 
        img.save('%s%s.png'%(resized_path,name)) # 이미지 저장 

        print(name+'  '+str(index) + '/' +str(total_image))
        index = index+1
