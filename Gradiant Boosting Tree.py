# Gradiant Boosting Tree
- 이전 트리를 점차 학습하는 형태
- learnin_rate로 정해진 학습률에 따라 오분류 데이터 포인트에 더 높은 가중치 부여,
- 다음에 생성되는 트리는 오분류 데이터의 올바른 분류에 더 초점을 맞추는 형식
- 복잡도가 낮은 초기 트리 모델로부터 점차 복잡해지는 형태를 갖춤
- 랜덤포레스트보다 더 적은수의 트리로도 높은 예측력을 기대할 수 있음
- 각 트리는 서로 독립적일 수 없으므로 n_jobs같은 parallel 옵션에 대한 기대가 줄어든다
from sklearn.ensemble import GradientBoostingClassifier as gb
from sklearn.ensemble import GradientBoostingRegressor as gb_r

# [Ref. XGB install and loading]
# pip install xgboost
from xgboost.sklearn import XGBClassifier as xgb
from xgboost.sklearn import XGBRegressor as xgb_r

# 1. Model creation and learning 
df_cancer = cancer()
x_train, x_test, y_train, y_test = train_test_split(df_cancer['data'],
                                                    df_cancer['target'],
                                                    random_state=99)

m_gb = gb()
m_gb.fit(x_train,y_train)
# n_estimators = 100, max_depth=3, learning_rate=0.1, min_samples_split=2

# 2. Model evaluation
m_gb.score(x_test, y_test)   # 95 %

# 3. Parameter tunning

score_train=[]; score_test=[]
for i in [0.001,0.01,0.1,0.5,1]:
    m_gb = gb(learning_rate=i)
    m_gb.fit(x_train, y_train)
    score_train.append(m_gb.score(x_train, y_train))
    score_test.append(m_gb.score(x_test,y_test))
    
plt.plot([0.001,0.01,0.1,0.5,1], score_train, label='train_score')
plt.plot([0.001,0.01,0.1,0.5,1], score_test, label='test_score')
plt.legend()

# Property visualization
m_gb = gb(learning_rate = 0.1)
m_gb.fit(x_train, y_train)

plot_feature_importances_cancer(m_gb, df_cancer)
