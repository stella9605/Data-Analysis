# 회귀 분석 : 특정변수(독립변수)가 다른 변수(종속변수)에 어떠한 영향을 미치는가 (인과관계분석)

# Decision Tree > Random Forest > Gradiant Boosting Tree(GB)
#                               > extreme Gradiant Boosting Tree(XGB)
                              

# max_depth : 설명변수의 재사용 횟수
# max_features : 각 노드에서 설명변수 선택시 고려되는 후보의 개수
#              : 값이 작을수록 서로 다른 트리가 구성될 확률이 높다
# min_samples_split : 최소 가지치기 개수 ( 오분류 건수가 해당값 이상일 경우 추가 split)
#                   : 값이 작을수록 모델이 복잡해짐 

# 1. Data load
from sklearn.datasets import load_breast_cancer as cancer
df_cancer = cancer()

df_cancer.keys()
# dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])

# 2. train set, test set split

x_train, x_test, y_train, y_test = train_test_split(df_cancer['data'],
                                                    df_cancer['target'],
                                                    random_state=0)

# 3. Data learngin
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.tree import DecisionTreeRegressor as dt_r

m_dt = dt()
m_dt.fit(x_train, y_train)

# 4. Model evaluation

m_dt.score(x_test, y_test)  # 88% 

# 5. Parameter tunning 

score_train=[]; score_test=[]
for i in np.arange(2,21):
    m_dt = dt(min_samples_split=i)
    m_dt.fit(x_train, y_train)
    score_train.append(m_dt.score(x_train,y_train))
    score_test.append(m_dt.score(x_test,y_test))

plt.plot(np.arange(2,21), score_train, label='train_score')
plt.plot(np.arange(2,21), score_test, label = 'test_score')
plt.legend()


# 6. model Fixed
m_dt = dt(min_samples_split=11)
m_dt.fit(x_train, y_train)
m_dt.feature_importances_

s1 = Series(m_dt.feature_importances_, index = df_cancer.feature_names)
s1.sort_values(ascending=False)



