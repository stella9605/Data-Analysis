from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder


import pandas as pd
import numpy
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 입력
iris.features
df_iris=iris()
df = pd.read_csv('iris.csv', names = ["sepal.length", "sepal_width",
                                                 "petal_length","petal_width","species"])


df['sepal.length'] = float(df['sepal.length'])
# 데이터 분류
dataset = df.values
x = dataset[:,:4].astype(float)
y = dataset[:,4:]

# 문자열을 숫자로 변환
e = LabelEncoder()
e.fit(y)
y=e.transform(y)
y_encoded = np_utils.to_categorical(y)

# 모델의 설정 

model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu'))
model.add(Dense(4, activation='softmax'))

# 모델 컴파일
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 모델 실행
model.fit(x, y_encoded, epochs=50, batch_size=1)
