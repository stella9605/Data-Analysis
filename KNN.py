from sklearn.datasets import load_iris

df_iris = load_iris()

type(df_iris)

df_iris.keys()
# ['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename']

df_iris['data']
df_iris['target']
df_iris['target_names']
df_iris['DESCR']
df_iris['feature_names']
df_iris['filename']


# KNN 
#- 지도학습 > 분류분석 > 거리기반 모델
#- 예측하고자 하는 데이터와 기존 데이터포인트(각 관측치)들의 거리가 가까운
#   k개의 이웃이 갖는 정답(Y)의 평균 및 다수결로 Y를 예측하는 형태
#- 이상치(outlier)에 매우 민감 => 제거 혹은 수정 필요
#- 설명변수의 scale 에 매우 민감 => scale조정 필요(표준화)
#- 모델 학습시 선택된 설명변수의 조합에 따라 매우 다른 예측력을 보임
#  ( 모델 자체 feature selection 기능 없음)


# 1. Data classification

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x,                # Explanatory variable
                                                   y,                # Dependent variable
                                                   train_size=0.75,
                                                   random_state=0)

x_train, x_test, y_train, y_test = train_test_split(df_iris['data'],
                                                    df_iris['target'],
                                                    random_state=99)   # Fixed random values

x_train.shape    # (112,4), row of trunc(150 * 0.75)
x_test.shape     # (38,4)

# 2. Data learning
from sklearn.neighbors import KNeighborsClassifier as knn

m_knn = knn(n_neighbors=3)
m_knn.fit(x_train, y_train)

# 3. knn model evaluation

m_knn.score(x_test, y_test)  # 92

#  [Ref. How to get score without using the scores method ?]
m_knn.predict(x_test)   # Prediction result for test set
y_test                  # Actual answer to test set 



# 4. Parameter tunning

score_train=[]; score_test=[]
for i in np.arange(1,11):
    m_knn = knn(n_neighbors = i)
    m_knn.fit(x_train, y_train)
    score_train.append(m_knn.score(x_train, y_train))
    score_test.append(m_knn.score(x_test, y_test))
    
    
# As the value of K increases, the effect of outliers decreases,
# but the classification itself may not be possible
    
    
# 5. Visualization


import matplotlib.pyplot as plt
plt.plot(np.arange(1,11), score_train, label = 'train_score')
plt.plot(np.arange(1,11), score_test, label = 'test_score', color = 'red')
plt.legend()

# Prediction
new_data = np.array([5.5, 3.0, 2.5, 0.9])
new_data2 = np.array([[5.5, 3.0, 2.5, 0.9]])


m_knn.predict(new_data)
m_knn.predict(new_data2)

df_iris['target_names'][m_knn.predict(new_data2)[0]]
df_iris['target_names'][m_knn.predict(new_data2)][0]

