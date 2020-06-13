import pandas as pd
import numpy as np
from keras.models import Sequential 
from keras.layers.core import Dense 
import mglearn.plots.plot_scaling()

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data,
                                                    cancer.target,
                                                    random_state=100,
                                                    stratify = cancer.target )

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaler = scaler.fit(x_train)
x_train_scaler.transform(x_train)
# => x_train_scaler = scaler.fit_transform(x_train)


# scaler 는 fit과 transform 메서드를 지니고 있다.
# fit 메서드로 데이터 변환을 학습하고. transform 메서드로 실제 데이터의
# 스케일을 조정한다.

# fit 메서드는 학습용 데이터(train data)에만 적용해야 한다.
# 그 후, transform 메서드를 학습용 데이터와 테스트 데이터에 적용한다.(train data & test data)
# scaler는 fit_transform() 라는 단축메서드를 제공하고
# 학습용 데이터에는 fit_transform() 메서드를 적용하고,
# 테스트 데이터에는 fit_transform() 메서드를 적용한다

print('스케일 조정 전  features Min value : \n{}', format(x_train.min(axis=0)))
print('스케일 조정 전  features Max value : \n{}', format(x_train.max(axis=0)))
print('스케일 조정 후  features Min value : \n{}', format(x_train_scaler.min(axis=0)))
print('스케일 조정 후  features Min value : \n{}', format(x_train_scaler.min(axis=0)))

# RobustScaler
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
x_train_scaler = scaler.fit(x_train)
x_train_scaler.transform(x_train)

#MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train_scaler = scaler.fit(x_train)
x_train_scaler.transform(x_train)

# SVC로 cancer data set train

# 1. 데이터 스케일링을 적용하지 않은 상태
from sklearn.svm import SVC
x_train, x_test, y_train, y_test = train_test_split(cancer.data,
                                                    cancer.target,
                                                    random_state = 100)

svc = SVC()
svc.fit(x_train,y_train)             # svc 학습시키기                                     
svc.score(x_test,y_test)           # 점수 94% 


# 2. StandardScaler로 조정하고 SVC 학습시키기
scaler = StandardScaler()
x_train_scale = scaler.fit_transform(x_train)
x_test_scale = scaler.transform(x_test)
svc.fit(x_train_scale, y_train)   # fit은 학습용 데이터만 학습시킬수 있다. 
svc.score(x_test_scale, y_test)   # 97 실제데이터를 다루기 때문에 test로 score한다. 
