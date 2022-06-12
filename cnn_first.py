import tensorflow as tf
import numpy as np
import keras
import os
import glob
import img_np
import matplotlib.pyplot as plt # 그래프 관련 라이브러리
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
os.environ['TF_CFF_MIN_LOG_LEVEL'] = '3' # 텐서 플로우 오류 해결 코드 


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
    # data_dir = "D:\\image\\"
    batch_size = 120


    # data_load = tf.keras.processing.image_dataset_from_directory(
    #     data_dir = "D:\\resize_image\\",
    #     validation_split=0.2,
    #     subset="training",
    #     seed=123,
    #     image_size=(img_height, img_width),
    #     batch_size=batch_size
    # # )
    # np_list=[]
    # glob_data = r"D:\\numpy_array\\*.npy"
    # np_list = glob.glob(glob_data)
    img_np.image_to_numpy_array()
    directory_path = "C:\\Users\\SCHCsRC\\Desktop\\병열\\Python\\Machine-Learning-Classification\\numpy_array"
    file_list = os.listdir(directory_path)

    X_train = np.load(file_list)

    # X_train =  glob.glob('*.npy')
    Y_train = 0

    model = keras.Sequential() #구성
    model.add(keras.layers.Conv2D(32, kernel_size =3, strides=(1,1), padding="same", activation="relu", input_shape=(28, 28, 1))) # 3*3 * 64
    model.add(MaxPooling2D(pool_size=(1,1)))
    model.add(Conv2D(32, kernel_size =3, strides=(1,1),padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=(1,1)))
    model.add(Conv2D(32,kernel_size =3, strides=(1,1), padding="same", activation="relu"))
    model.add(Flatten())
    model.add(Dense(50, activation ='relu'))
    model.add(Dropout(0.5)) # test 1
    model.add(Dense(10, activation= 'sigmoid')) 
    model.summary() # 시각화 

    model.compile(loss='binary_crossaentropy', optimizer='adam', metrics='accuray') # 학습 
    history = model.fit(X_train, Y_train, epochs=20) #훈련
    # print('accuracy:', score[1])#정확도
    Performance_graph(history) # 그래프 
    print('학습 성공')
