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
    Image.MAX_IMAGE_PIXELS = None
    train_gen = ImageDataGenerator(horizontal_flip=True, rescale =1/255., rotation_range=30, shear_range=0.2, zoom_range=0.4)
    test_gen = ImageDataGenerator(rescale=1/255.)

    train_flow_gen = train_gen.flow_from_directory(directory='Asm_file\\resize\\train\\',
                                               target_size=(28,28),
                                               class_mode='binary',
                                               color_mode='grayscale',
                                               batch_size=10,
                                               shuffle=True)
    test_flow_gen = test_gen.flow_from_directory(directory = 'Asm_file\\resize\\test\\', 
                                             target_size = (28,28),
                                             class_mode='binary',
                                             color_mode='grayscale',
                                             batch_size = 10,
                                             shuffle=False)


    # opt = SGD(lr=0.01, momentum = 0.9, decay=0.01)
    model = tf.keras.Sequential() #  Sequential 모델 생성
    model.add(tf.keras.layers.Conv2D(input_shape=(28, 28, 1), kernel_size=(3, 3), filters=32, padding='same', activation='relu'))
    # 컨볼루션 층 추가 (입력데이터의 형태를 28x28x1로 지정, 합성곱에 사용되는 필터의 크기를 3x3으로 지정, 필터의 개수는 32개, 활성화 함수로 relu함수 사용)
    model.add(tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=64, padding='same', activation='relu'))
    # 컨볼루션 층 추가 (3x3 크기의 필터를 64개 이용, 활성화 함수로 relu 사용)
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    # 컨볼루션 층 추가 (최대 풀링에 2x2 크기의 필터를 사용)
    model.add(tf.keras.layers.Dropout(rate=0.5))
    # 0.5의 비율로 0으로 변환

    model.add(tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=128, padding='same', activation='relu'))
    # 컨볼루션 층 추가 (3x3 필터를 128개 이용, 활성화 함수로 relu 사용)
    model.add(tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=256, padding='valid', activation='relu'))
    # 컨볼루션 층 추가 (3x3 필터를 256개 이용, 활성화 함수로 relu 사용)
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    # 컨볼루션 층 추가 (최대 풀링에 2x2 크기의 필터를 사용)
    model.add(tf.keras.layers.Dropout(rate=0.5))
    # 0.5의 비율로 0으로 변환

    model.add(tf.keras.layers.Flatten())
    # 1차원 벡터로 변환
    model.add(tf.keras.layers.Dense(units=512, activation='relu'))
    # 은닉층에 해당, 뉴런의 수는 512개이고, relu 활성화 함수 사용
    model.add(tf.keras.layers.Dropout(rate=0.5))
    # 0.5의 비율로 0으로 변환
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    # 은닉층에 해당, 뉴런의 수는 256개이고, relu 활성화 함수 사용
    model.add(tf.keras.layers.Dropout(rate=0.5))
    # 0.5의 비율로 0으로 변환
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) 
    # 출력층에 해당, 따라서 노드 수는 한 개로 지정

    # model = Sequential()
    # model.add(Conv2D(32, (3, 3),activation='relu', strides=(1,1), padding="same",input_shape=(64,64,1)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(64, (3, 3), activation='relu', strides=(1,1), padding="same"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(128, (3, 3), activation='relu', strides=(1,1), padding="same"))
    # model.add(Dense(256, activation='relu'))
    # model.add(Flatten())
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='sigmoid'))
    model.summary() # 모델의 구조 요약 출력
    model.compile(loss='MSE',optimizer='adam', metrics=['accuracy']) # 모델 학습을 위한 학습 방식 설정
    # cm11 = confusion_matrix(y11_test, ysvc11_pred)
    # print(cm11)
    # print(metrics.classification_report(y11_test, ysvc11_pred))
    train_hist = model.fit(train_flow_gen, epochs =20) # 설정한 학습방식을 기준으로 학습 진행
    test_hist = model.evaluate(test_flow_gen) # 모델 평가
    Performance_graph(train_hist) # 모델 그래프 생성
    
    