import array
import cv2
import os
import numpy as np
import codecs
import glob



#데이터 이미지로 변환 
def convert_asm_to_images(sourcepath, destpath):
    files = os.listdir(sourcepath) # 모든 파일과 디렉토리 리턴 
    print('Sourcepath :', sourcepath)
    print('Destination path :', destpath)
    count =1
    for file in files:
        if file.endswith(".asm"): # 확장자 검색 
            f = codecs.open(sourcepath+file, 'rb') # 읽기 모드로 오픈(바이너리 모드)
            Asm_Data_Size = os.path.getsize(sourcepath+file)
            width = int(Asm_Data_Size**0.5) # 크기 수정 
            print(width)
            byte_list = array.array("B") # 바이트 요소 담기 위한 바이트 배열 선언 
            byte_list.frombytes(f.read()) 
            f.close()
            Write_Data = np.reshape(byte_list[:width * width], (width, width)) # 2차원 배열 
            Write_Data = np.uint8(Write_Data) # 픽셀 당 바이트 수 
            cv2.imwrite(destpath+file+'.png', Write_Data) # 이미지 파일 저장 
            print(str(count) +'번째 :' + file)
            count += 1
    print('이미징 작업 완료')

# 시작 
def convert_start():
    sourcepath = r"D:\disasmbled\train__malware\\"
    destpath = r"D:\image\\"
    convert_asm_to_images(sourcepath, destpath)