import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt # 그래프 관련 라이브러리


def Performance_graph(history): # history 인자 파악 
    plt.figure(figsize=(12,5))#최초창크기
    plt.subplot(1,2,1) #그래프 위치 
    plt.plot(history.history['accuracy'])
    plt.title('model accuray') # 그래프 title 설정 
    plt.xlabel('epoch')
    plt.ylabel('accuracy') 
    plt.legend(['train', 'validation'], loc='lower right') #범례
    plt.show()
  # x축 레이블y설정
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'validation'],  loc='lower right')
    plt.show()

img_rows = 255
img_cols = 255

(x_train, y_train), (x_test, y_test) = #읽을데이타 라벨링 작업 추가 이거 찿아봤는데 이해를 못함 
img_cols, img_rows = x_train.shape[1: ex)255] # 이게 첫번째 데이터부터 예를들어 255번째 데이터까지 불러오는 로직


model = Sequential() #구성
model.add(Conv2D(64,(3, 3), stride=(1,1), padding="same", input_shape=(255, 255, 1))) # 3*3 * 64
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3, 3), stride=(1,1), padding="same"))
 # ㄱ그글그래랲래프 x, yㅊ축 값 설정 model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) #층추가
model.add(dense(100)) # dense층 개수
model.add(Activation('sigmoid')) # 그래프 화면에 출력
model.summary() #구성한 신경망 정보 출력

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuray']) #훈련 뭐야 이거 학습 시키는거? 근데 시발 저거 라벨링을 어떻게 하냐 
train_history = model.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test)) #epochs 전체반복주기 #batch size

score= model.evaluate(x_test, y_test, verbose=1) #예측 verbos=1일 때만 과정이 나옴
print('loss:',score[0])#손실
print('accuracy:', score[1])#정확도
Performance_graph(train_history) # 그래프 




# 여기까지가 세준이가 한거

import tensorflow as tf
import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

def feature():
    sourcepath = 'D:\\image\\'
    IMG_HEIGHT = IMG_WIDTH = 255 # 픽셀 수 

    batch_size = null # 데이터 셋 쪼개는 단위 

    # 이미지 데이터 불러오기
    files = os.listdir(sourcepath)
    # 이미지 데이터 컨볼루션 진행
    (x_train, y_train), (x_test, y_test) =  # x_train = 라벨링 작업 진행 

    input_shape = (255, 255, 1) # grey-scale

    # 모델화(이미지는 2차원 배열인 따라서 Conv2D 사용)
    # 채널 수 1
    model = Sequential()
    model.add(Conv2D(filters=, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu", input_shape=())) # 보통 필터 3x3
    model.add(MaxPooling2D(pool_size=, strides=()))
    model.add(Conv2D(filter=, kernel_size=(3,3), strides(1,1), 입력 형태))
    model.add(MaxPooling2D(pool_size=(), strides=()))
    ...

    model.add(Dropout(입력 값)) # 과적합 방지를 위해 무작위로 특정 노드(입력 값)를 0으로 만드는 함수.
    model.add(Flatten()) # 2차원데이터를 1차원 데이터로 변환
    model.add(Dense(출력 뉴런, 입력 뉴런 수, 활성화 함수)) # 
   요약