run profile1
import math
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.neighbors import KNeighborsRegressor as knn_r

from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.tree import DecisionTreeRegressor as dt_r

from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import RandomForestRegressor as rf_r

from sklearn.ensemble import GradientBoostingClassifier as gb
from sklearn.ensemble import GradientBoostingRegressor as gb_r

from sklearn.model_selection import RandomizedSearchCV


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

X = train.iloc[:, :-4]
Y = train.iloc[:,-4:]

x_train, x_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    random_state=0)


Y_1 = Y.iloc[:,0:1]  # hhb
Y_2 = Y.iloc[:,1:2]  # hbo2
Y_3 = Y.iloc[:,2:3]  # ca
Y_4 = Y.iloc[:,3:4]  # na

# hbb,hbo2 - train, test set
# hhb
x_train_hhb, x_test_hhb, y_train_hhb, y_test_hhb = train_test_split(X,
                                                                    Y_1,
                                                                    random_state = 0)


# 흡광도 
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
    
tuning_x_train =  x_train_hhb.apply(tuning_var, axis=1) # 7500 x 35
tuning_x_test = x_test_hhb.apply(tuning_var, axis =1)   # 2500 x 35



# RandomForestRegressor
m_rf = rf_r()
model = m_rf.fit(tuning_x_train,y_train_hhb.values.ravel())
m_rf.score(tuning_x_train,y_train_hhb.values.ravel())   # 0.92

# min_samples_leaf=1, min_samples_split=2,
# n_estimators=10,max_features='auto'

# 매개변수 튜닝
# n_estimators = 3 
score_train=[]; score_test=[]
for i in np.arange(1,100):
    m_rf = rf_r(n_estimators=i)
    m_rf.fit(tuning_x_train,y_train_hhb.values.ravel())
    score_train.append(m_rf.score(tuning_x_train,y_train_hhb.values.ravel()))
    score_test.append(m_rf.score(tuning_x_test,y_test_hhb.values.ravel()))
    
    
plt.plot(np.arange(1,101), score_train, label='train_score')
plt.plot(np.arange(1,101), score_test, label='test_score', color='red')
plt.legend() 

# min_samples_split 
score_train=[]; score_test=[]
for i in np.arange(2,51):
    m_rf = rf_r(min_samples_split=i)
    m_rf.fit(tuning_x_train,y_train_hhb.values.ravel())
    score_train.append(m_rf.score(tuning_x_train,y_train_hhb.values.ravel()))
    score_test.append(m_rf.score(tuning_x_test,y_test_hhb.values.ravel()))
    
    
plt.plot(np.arange(1,101), score_train, label='train_score')
plt.plot(np.arange(1,101), score_test, label='test_score', color='red')
plt.legend() 

# max_features
score_train=[] ; score_test=[]
for i in np.arange(1,tuning_x_train.shape[1] + 1) :
    m_rf = rf_r(max_features=i)
    m_rf.fit(tuning_x_train, y_train_hhb.values.ravel())
    score_train.append(m_rf.score(tuning_x_train, y_train_hhb.values.ravel()))
    score_test.append(m_rf.score(tuning_x_test, y_test_hhb.values.ravel()))


plt.plot(np.arange(1,31), score_train, label='train_score')
plt.plot(np.arange(1,31), score_test, label='test_score', color='red')
plt.legend() 



# 모델 고정
m_rf = rf_r(n_estimators=11, min_samples_split = 26, max_features = 10)
m_rf.fit(tuning_x_train, y_train_hhb)
m_rf.score(tuning_x_train,y_train_hhb.values.ravel())   # 0.79




