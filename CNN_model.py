from xml.etree.ElementInclude import include
from keras.applications import Xception
from keras.layers import Input, Conv2D, Dense, Dropout, Flatten, BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import array
import cv2
import os
import codecs



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
    if not os.path.isdir('resize'):
        os.mkdir('resize')
    if not os.path.isdir('resize\\train'):
        os.mkdir('resize\\train')
    if not os.path.isdir('resize\\test'):
        os.mkdir('resize\\test')
    # print('생성')
    # 모든 데이터셋에 대한 경로 지정 

    Dir_train_benign_path = 'train__benign\\'
    Dir_train_malware_path = 'train__malware\\'
    Dir_test_benign_path ='test_benign\\'
    Dir_test_malware_path='test_malware\\'
    add_train_benign_path = "resize\\train\\benign\\"
    add_train_malware_path = "resize\\train\\malware\\"
    add_test_benign_path = "resize\\test\\benign\\"
    add_test_malware_path = "resize\\test\\malware\\"
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
        convert_asm_to_images(Dir_test_malware_path, add_test_malware_path)
    else:
        os.mkdir(add_test_malware_path) # 디렉토
        convert_asm_to_images(Dir_test_malware_path, add_test_malware_path)# 이미지 크기 조절 

def Performance_graph(history): # history 인자 파악 
    plt.figure(figsize=(12,4)) # 최초창크기를 가로 12인치 세로 4인치로 설정
    plt.subplot(1,2,1) # 그래프 위치 
    plt.plot(history.history['accuracy']) # 훈련과정의 시각화(정확도)
    plt.title('model accuray') # 그래프의 title 설정 
    plt.xlabel('epoch') # 그래프의 x축 title 지정
    plt.ylabel('accuracy') # 그래프의 y축 title 지정
    plt.legend(['train', 'validation'], loc='lower right') # 범례 (그래프 우측 하단에 데이터의 종류 표시)
    plt.show() # 그래프 생성
  # x축 레이블y설정
    plt.subplot(1,2,2) # 그래프 위치
    plt.plot(history.history['loss']) # 훈련과정의 시각화(손실)
    plt.title('model loss') # 그래프의 title 설정
    plt.xlabel('epoch') # 그래프의 x축 title 지정
    plt.ylabel('loss') # 그래프의 y축 title 지정
    plt.legend(['train', 'validation'],  loc='lower right') # 범례 (그래프 우측 하단에 데이터의 종류 표시)
    plt.show() # 그래프 생성

# 이미징 데이터 생성기 설정
def model_ImageData_Gen_training():
    Image.MAX_IMAGE_PIXELS = None #최대 픽셀을 넘을 경우 학습 중단되는 것 방지 
    #Generator 생성 
    train_gen = ImageDataGenerator(horizontal_flip=True, 
                                   rescale =1/255., # 0~1 숫자가 나오도록 
                                   rotation_range=30, 
                                   shear_range=0.2,
                                   zoom_range=0.4)
    test_gen = ImageDataGenerator(rescale=1/255.)

    train_flow_gen = train_gen.flow_from_directory(directory='resize\\train\\',
                                               target_size=(64,64), # 사이즈 재 조정 
                                               class_mode='binary', # 분류 모드 = 이진
                                               color_mode='grayscale', # 흑백 이미지
                                               batch_size=40, # 배치 사이즈 
                                               shuffle=True) # 셔플 여부 
    test_flow_gen = test_gen.flow_from_directory(directory = 'resize\\test\\', 
                                             target_size = (64,64),
                                             class_mode='binary',
                                             color_mode='grayscale',
                                             batch_size = 40,
                                             shuffle=False)
    # 모델
    model = Sequential()
    model.add(Conv2D(32, (3, 3),activation='relu', strides=(1,1), padding="same",input_shape=(64,64,1))) 
    # 컨볼루션 층 추가 (입력데이터의 형태를 64x64x1로 지정, 합성곱에 사용되는 필터의 크기를 3x3으로 지정, 필터의 개수는 32개, 활성화 함수로 relu함수 사용)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 풀링 층 추가 (최대 풀링에 2x2 크기의 필터를 사용)
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1,1), padding="same"))
    # 컨볼루션 층 추가 (3x3 크기의 필터를 64개 이용, 활성화 함수로 relu 사용)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1,1), padding="same"))
    # 컨볼루션 층 추가 (3x3 크기의 필터를 128개 이용, 활성화 함수로 relu 사용)
    model.add(Dense(256, activation='relu'))
    # 은닉층에 해당, 뉴런의 수는 512개이고, relu 활성화 함수 사용
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    #노드를 0.5의 비율로 제거 
    model.add(Dense(1, activation='sigmoid')) 
    # 이진 분류 모델이기 때문에 출력은 1로 지정 
    model.summary() # 모델의 구조 요약 출력
    model.compile(loss='MSE',optimizer='adam', metrics=['accuracy']) # 모델 학습을 위한 학습 방식 설정
  
    train_hist = model.fit(train_flow_gen, epochs =40) # 설정한 학습방식을 기준으로 학습 진행
    test_hist = model.evaluate(test_flow_gen) # 모델 평가
    Performance_graph(train_hist) # 모델 그래프 생성

if __name__=="__main__":
    # 이미지 변환 수행 
    convert_start() 
    
    # 이미지 전처리 및 모델 생성, 학습 진행 
    model_ImageData_Gen_training()
 
