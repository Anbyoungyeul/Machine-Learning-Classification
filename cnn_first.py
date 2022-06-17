import tensorflow as tf
import numpy as np
import keras
import os
import matplotlib.pyplot as plt # 그래프 관련 라이브러리
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D

def Performance_graph(history): # history 인자 파악 
    plt.figure(figsize=(12,4))#최초창크기
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

def model_learning(): # 이미지 
    img_height = 28
    img_width = 28 # 28*28 사이즈
    data_dir = "D:\\image\\"
    batch_size = 120


    data_load = tf.keras.processing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    X_train, X_test, Y_train, Y_test = np.load('./img_np.py') 

    model = Sequential() #구성
    model.add(Conv2D(64,(3, 3), stride=(1,1), padding="same", activation="relu", input_shape=(28, 28, 1))) # 3*3 * 64
    model.add(MaxPooling2D(pool_size=(1,1)))
    model.add(Conv2D(64,(3, 3), stride=(1,1), padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=(1,1)))
    model.add(Conv2D(64,(3, 3), stride=(1,1), padding="same", activation="relu"))
 # 그래프  x, y축 값 설정 model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))#과적합 해결  학습할 떈 0.5, 테스트 1
    model.add(Flatten()) # 평탄화 층
    model.add(Dense(100)) # dense층 개수
    model.add(Activation('sigmoid')) # 시그모이드 활성화 출력 함수 
    model.summary() #구성한 신경망 정보 출력  

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuray']) # 학습 
    train_history = model.fit(x_train, y_train, batch_size=128, epochs=5, validation_data=(x_test, y_test)) #epochs 전체반복주기 #batch size

    score= model.evaluate(x_test, y_test, verbose=1) #예측 verbos=1일 때만 과정이 나옴
    print('loss:',score[0])#손실
    print('accuracy:', score[1])#정확도
    Performance_graph(train_history) # 그래프 


