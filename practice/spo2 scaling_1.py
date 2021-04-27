import pandas as pd
import numpy as np
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict
import math
from keras.models import Sequential 
from keras.layers.core import Dense 
from keras.utils import np_utils 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
run profile1


train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)

train.head()

train.isnull().sum()[train.isnull().sum().values > 0]

# 결측치 보완
train_dst = train.filter(regex='_dst$', axis=1).replace(0, np.NaN) # dst 데이터만 따로 뺀다.
test_dst = test.filter(regex='_dst$', axis=1).replace(0, np.NaN) # 보간을 하기위해 결측값을 삭제한다.
test_dst.head(1)

train_dst = train_dst.interpolate(methods='quadratic', axis=1)
test_dst = test_dst.interpolate(methods='quadratic', axis=1)

# 스팩트럼 데이터에서 보간이 되지 않은 값은 0으로 일괄 처리한다.
train_dst.fillna(0, inplace=True) 
test_dst.fillna(0, inplace=True)

test_dst.head(1)

train.update(train_dst) # 보간한 데이터를 기존 데이터프레임에 업데이트 한다.
test.update(test_dst)


X = train
Y = train.iloc[:,-4:]

Y_hhb = Y.iloc[:,0]
Y_hbo2 = Y.iloc[:,1]
Y_ca = Y.iloc[:,2]
Y_na = Y.iloc[:,3]

# inf 제거 
# spo2 = spo2[~spo2.isin([np.nan, np.inf, -np.inf]).any(1)]
# -- 제거하니 행의 개수가 안맞음

# inf -1 으로 치환
d_spo2 = d_spo2.replace([np.inf, -np.inf], np.nan).fillna(-1)   # data frame 형식
spo2 = spo2.replace([np.inf, -np.inf], np.nan).fillna(-1)       # series 형식 


# 데이터셋 분리
x_train, x_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    random_state=99)

X.iloc[:,-4]
X.iloc[:,-3]

# 튜닝함수 -spo2
def tuning_spo2(s):
    # 계산식(산소포화도)
    s_hhb = s.iloc[:,-4]    # hhb
    s_hbo2 = s.iloc[:,-3]    # hbo2
     
    # # index 표준화
    # set_index = s.index
    # s_hhb.index = set_index
    # s_hbo2.index= set_index
    
    spo2 = s_hbo2 / (s_hbo2 + s_hhb)
    spo2 = spo2.replace([np.inf, -np.inf], np.nan).fillna(-1) 
    return(spo2)
    s_hbo2.index = set_index
     
    spo2_t = s_hbo2 / (s_hhb + s_hbo2)
    spo2_t = spo2_t.replace([np.inf, -np.inf], np.nan).fillna(0)
    spo2_t.index = set_index
    return (spo2_t)

tuning_spo2 = tuning_spo2(X)

tuning_spo2 = DataFrame(tuning_spo2)

tuning_spo2.index = X.index

tuning_x_train = x_train.apply(tuning_spo2, axis=1)
tuning_y_test = y_test.apply(tuning_spo2, axis=1)


Y_hhb
Y_hbo2

Y_hbo2 / Y_hhb + Y_hbo2

X[1:36].columns.str.split('_').str[0]
X
Y

def tuning_var(s):
    s_rho = s[0]          # _rho
    s_src = s[1:36]       # _src
    s_dst = s[36:]        # _dst    

    # index 표준화
    set_index = s_src.index.str.split('_').str[0]
    s_src.index = set_index
    s_dst.index = set_index

    # 계산식 (흡광도 계산식)
    # A(흡광도) = -log10(I(투과방사선)/I0(입사방사선))  
    #           = ε(흡광계수) ⋅ b(투과 경로 길이(cm)) ⋅ c(농도)
    s_ds_st = ((s_dst / s_src) / (s_rho/10))
    
    # 계산 완료후 inf,nan 0으로 치환
    s_ds_st = [i if i != np.inf else 0.0 for i in s_ds_st ]
    s_ds_st = Series(s_ds_st).fillna(value = 0)
    
    # math.log 계산을 위해 0을 1로 치환후 계산(흡광계수는 1로 가정한다.)
    s_ds_st = [1 if i == 0 else i for i in s_ds_st ]
    
    # 변수 튜닝 반환
    out_s = Series(map(lambda x : -math.log(x,10), s_ds_st))
    out_s.index= set_index
    return(out_s)

tuning_x_train = x_train.apply(tuning_var, axis=1)
tuning_x_test = x_test.apply(tuning_var)




# 스케일링 
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)

# 모델 구성 
model = Sequential()
model.add(Dense(32, input_dim=x_train_scaler.shape[1], activation='relu'))
model.add(Dense(8))
model.add(Dense(1))


model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['MAE']) 

# 모델 학습 
hist = model.fit(x_train_scaler, y_train, epochs = 100, batch_size = 100) # loss 2.39, MAE 1.00

# 모델 평가 
model.evaluate(x_test_scaler, y_test)   # 1.07

# 모델 사용
xhat = x_test[0:1]
model.predict(xhat)  # 5.96 


# standard model.fit => loss:2.39, MAE:1.00, model.evaluate:1.07, model.predict:5.96
# Robuster predict 1.60



tunning_x_train =  x_train.apply(tunning_spo2, axis=1)
tunning_x_train.index



# 산소포화도에 흡광도 학습
x_train, x_test, y_train, y_test = train_test_split(X,
                                                    d_spo2,
                                                    random_state=99)

x_train = x_train.astype('float64')
x_test = x_test.astype('float64')

def tuning_var(s):
    s_rho = s[0]          # _rho
    s_src = s[1:36]       # _src
    s_dst = s[36:]        # _dst    

    # index 표준화
    set_index = s_src.index.str.split('_').str[0]
    s_src.index = set_index
    s_dst.index = set_index

    # 계산식 (흡광도 계산식)
    # A(흡광도) = -log10(I(투과방사선)/I0(입사방사선))  
    #           = ε(흡광계수) ⋅ b(투과 경로 길이(cm)) ⋅ c(농도)
    s_ds_st = ((s_dst / s_src) / (s_rho/10))
    
    # 계산 완료후 inf,nan 0으로 치환
    s_ds_st = [i if i != np.inf else 0.0 for i in s_ds_st ]
    s_ds_st = Series(s_ds_st).fillna(value = 0)
    
    # math.log 계산을 위해 0을 1로 치환후 계산(흡광계수는 1로 가정한다.)
    s_ds_st = [1 if i == 0 else i for i in s_ds_st ]
    
    # 변수 튜닝 반환
    out_s = Series(map(lambda x : -math.log(x,10), s_ds_st))
    out_s.index= set_index
    return(out_s)

# 튜닝 변수
tuning_x_train = x_train.apply(tuning_var, axis=1) # 7500 x 35
tuning_x_test = x_test.apply(tuning_var, axis=1)   # 2500 x 35

# 튜닝 변수 스케일링

scaler = StandardScaler()

tuning_x_train_scaler = scaler.fit_transform(tuning_x_train)
tuning_x_test_scaler = scaler.transform(tuning_x_test)


tuning_x_train.shape[1]   #(7500, 35)
x_train.shape[1]          #(7500, 71)

# 모델 설정
model = Sequential()
model.add(Dense(35, input_dim = x_train.shape[1], activation = 'relu'))
model.add(Dense(10))
model.add(Dense(1))
 

# 모델 학습 과정 설정
model.compile(loss='mean_absolute_error',
              optimizer='adam',
              metrics=['mae'])

import tensorflow as tf
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01),
              lostt=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalCrossentropy])

# 모델 학습 시키기
hist = model.fit(x_train, y_train, epochs = 1000, batch_size = 300)

#
model.fit(x_test, y_test, epochs=1000, batch_size=300)

# 모델 평가하기 

#
model.evaluate(x_train,y_train,batch_size = 100 )[1]


model.evaluate(x_test, y_test, batch_size = 100)[1]



loss_and_matrics = model.evaluate(x_test, y_test, batch_size = 100)[1]
print(loss_and_matrics)  # 1.028




from keras.models import load_model

model.save = model('model_spo2.h5')

test1 = x_test[0:1]
model.predict(test1)



 
