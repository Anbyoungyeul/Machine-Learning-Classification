import tensorflow as tf
import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

IMG_HEIGHT = IMG_WIDTH = 255 # 픽셀 수 

batch_size = null # 데이터 셋 쪼개는 단위 

# 이미지 데이터 불러오기
files = os.listdir()
# 이미지 데이터 컨볼루션 진행
(x_train, y_train), (x_test, y_test) =  # x_train = 라벨링 작업 진행 

input_shape = (255, 255, 1) # grey-scale

# 모델화(이미지는 2차원 배열인 따라서 Conv2D 사용)
 model = Sequential()
 model.add(Conv2D(필터수, 커널 크기, 활성화 함수, 입력 형태))
 model.add(MaxPooling2D(풀링 사이즈, 스트라이드))
 model.add(Conv2D(필터수, 커널 크기, 활성화 함수, 입력 형태))
 model.add(MaxPooling2D(풀링 사이즈, 스트라이드))
 ...

 model.add(Dropout(입력 값)) # 과적합 방지를 위해 무작위로 특정 노드(입력 값)를 0으로 만드는 함수.
 model.add(Flatten()) # 2차원데이터를 1차원 데이터로 변환
 model.add(Dense(출력 뉴런, 입력 뉴런 수, 활성화 함수)) # 
 model.summary() # 모델 정보 요약