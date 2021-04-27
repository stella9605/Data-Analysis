# Random Forest - cancer data set 
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import RandomForestRegressor as rf_r


# 1. Model creation and learning 
from sklearn.datasets import load_breast_cancer as cancer

m_rf = rf()
x_train, x_test, y_train, y_test = train_test_split(df_cancer['data'],
                                                    df_cancer['target'],
                                                    random_state=99)

m_rf.fit(x_train, y_train)

# n_estimators = 100, max_features = 'auto', min_samples_split=2

# 2. Model evaluation
m_rf.score(x_test, y_test)  # 95%

# 3. Parameter tunning

# 3-1) Select number of trees

score_train=[]; score_test=[]
for i in np.arange(1,101):
    m_rf=rf(n_estimators=i)
    m_rf.fit(x_train,y_train)
    score_train.append(m_rf.score(x_train,y_train))
    score_test.append(m_rf.score(x_test,y_test))
    
plt.plot(np.arange(1,101), score_train, label='train_score')
plt.plot(np.arange(1,101), score_test, label='test_score', color='red')
plt.legend()

# 3-2) Select number of split 
score_train1=[]; score_test1=[]
for i in np.arange(2,21):
    m_rf=rf(min_samples_split = i)
    m_rf.fit(x_train, y_train)
    score_train1.append(m_rf.score(x_train, y_train))
    score_test1.append(m_rf.score(x_test, y_test))
    

plt.plot(np.arange(2,21), score_train1, label = 'train_score')
plt.plot(np.arange(2,21), score_test1, label='test_score', color ='red')
plt.legend()


# 3-3) SElect number of feature
score_train2=[]; score_test2=[]
for i in np.arange(1, df_cancer['data'].shape[1]+1):
    m_rf = rf(max_features=i)
    m_rf.fit(x_train, y_train)
    score_train2.append(m_rf.score(x_train, y_train))
    score_test2.append(m_rf.score(x_test, y_test))
    
plt.plot(np.arange(1,31), score_train2, label = 'train_score')
plt.plot(np.arange(1,31), score_test2, label='test_score', color='red')
plt.legend()


# 4. Final Model Fixation
m_rf = rf(n_estimators = 22, min_samples_split = 13, max_features=4)
m_rf.fit(x_train, y_train)
m_rf.score(x_test, y_test)   # 97%


# 5. Attribute Importance Visualization
def plot_feature_importances_cancer(model, data):
    n_features = data.data.shape[1]
    plt.barh(range(n_features), data.feature_names)
    plt.xlabel("Attribute Importance")
    plt.ylabel("Attribute")
    plt.ylim(-1, n_features)

plt.rc('font', family='Malgun Gothic')
plot_feature_importances_cancer(m_rf, df_cancer)

