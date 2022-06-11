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




# 여기까지가 세준이가 한거

# import tensorflow as tf
# import os
# import numpy as np
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers.convolutional import Conv2D, MaxPooling2D

# def feature():
#     sourcepath = 'D:\\image\\'
#     IMG_HEIGHT = IMG_WIDTH = 255 # 픽셀 수 

#     batch_size = null # 데이터 셋 쪼개는 단위 

#     # 이미지 데이터 불러오기
#     files = os.listdir(sourcepath)
#     # 이미지 데이터 컨볼루션 진행
#     (x_train, y_train), (x_test, y_test) =  # x_train = 라벨링 작업 진행 

#     input_shape = (255, 255, 1) # grey-scale

#     # 모델화(이미지는 2차원 배열인 따라서 Conv2D 사용)
#     # 채널 수 1
#     model = Sequential()
#     model.add(Conv2D(filters=, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu", input_shape=())) # 보통 필터 3x3
#     model.add(MaxPooling2D(pool_size=, strides=()))
#     model.add(Conv2D(filter=, kernel_size=(3,3), strides(1,1), 입력 형태))
#     model.add(MaxPooling2D(pool_size=(), strides=()))
#     ...

#     model.add(Dropout(입력 값)) # 과적합 방지를 위해 무작위로 특정 노드(입력 값)를 0으로 만드는 함수.
#     model.add(Flatten()) # 2차원데이터를 1차원 데이터로 변환
#     model.add(Dense(출력 뉴런, 입력 뉴런 수, 활성화 함수)) # 
#    요약