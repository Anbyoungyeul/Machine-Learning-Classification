import array
import cv2
import os
import numpy as np
import codecs
from PIL import Image

#데이터 이미지로 변환 
def convert_asm_to_images(sourcepath, add_path):
    files = os.listdir(sourcepath) # 모든 파일과 디렉토리 리턴 
    for file in files: # 디렉토리에 포함된 모든 파일에 대해 이미지 변환 수행
        if file.endswith(".asm"): # 확장자 검색 (파일이 .asm으로 끝나는 모든 파일에 대하여 수행)
            f = codecs.open(sourcepath+file, 'rb') # 해당 파일을 읽기 모드로 오픈(바이너리 모드)
            Asm_Data_Size = os.path.getsize(sourcepath+file) # 파일의 사이즈 확보
            width = int(Asm_Data_Size**0.5) # 확보된 파일의 사이즈를 0.5배로 크기 수정 
            byte_list = array.array("B") # 바이트 요소 담기 위한 바이트 배열 선언  
            byte_list.frombytes(f.read()) 
            f.close() # 파일 닫기
            Write_Data = np.reshape(byte_list[:width * width], (width, width)) # 바이트 배열을 2차원 배열로 변환
            Write_Data = np.uint8(Write_Data) # 픽셀 당 바이트 수 
            cv2.imwrite(add_path+file+'.png', Write_Data) # 이미지 파일 저장 
    print('이미징 작업 완료') # 이미징 작업이 성공적으로 완료되면 문자열 출력



# 이미징 변환 조건문
def convert_start():
    # 모든 데이터셋에 대한 경로 지정 
    Dir_train_benign_path = 'Asm_file\\training\\benign\\'
    Dir_train_malware_path = 'Asm_file\\training\\malware\\'
    Dir_test_benign_path ='Asm_file\\test\\benign\\'
    Dir_test_malware_path='Asm_file\\test\\malware\\'
    add_train_benign_path = "Asm_file\\resize\\train\\benign\\"
    add_train_malware_path = "Asm_file\\resize\\train\\malware\\"
    add_test_benign_path = "Asm_file\\resize\\test\\benign\\"
    add_test_malware_path = "Asm_file\\resize\\test\\malware\\"
    # 이미지 변환 경로 검사, 존재하지 않을 시 추가 후 변환 수행  
    if os.path.isdir(add_train_benign_path):
        convert_asm_to_images(Dir_train_benign_path, add_train_benign_path)
        # 첫번째 인자로 들어온 디렉토리의 파일을 두 번째 인자로 들어온 디렉토리에 파일을 이미지로 변환하여 저장
    else:
        os.mkdir(add_train_benign_path) #디렉토리
        convert_asm_to_images(Dir_train_benign_path, add_train_benign_path)
        
    if os.path.isdir(add_train_malware_path):
        convert_asm_to_images(Dir_train_malware_path, add_train_malware_path)
    else:
        os.mkdir(add_train_malware_path) # 
        convert_asm_to_images(Dir_train_malware_path, add_train_malware_path)
        
    if os.path.isdir(add_test_benign_path):
        convert_asm_to_images(Dir_test_benign_path, add_test_benign_path)
    else:
        os.mkdir(add_test_benign_path) # 디렉토
        convert_asm_to_images(Dir_test_benign_path, add_test_benign_path)
            
    if os.path.isdir(add_test_malware_path):
        convert_asm_to_images(Dir_test_malware_path, add_test_malware_path)# 이미지 크기 조절 

    
