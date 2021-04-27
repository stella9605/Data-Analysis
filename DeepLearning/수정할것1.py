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
import matplotlib.pyplot as plt

import warnings 
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Dense, Activation


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



# 각 조직에 관환 감쇠계수, 감쇠길이
# 감쇠 계수 DataFrame
attenuation_coefficient = pd.DataFrame({'NIR-1' : [10,48,25], 
                                        'NIR-2' : [7.4,45,23],
                                        'SWIR' : [6.5,44,21],
                                        'SWIR-2' : [9,49,24.5]})
# 감쇠 계수 index설정 (뇌 피질, 두개골, 피부)
attenuation_coefficient.index = ['Brain_cortex','Cranial_bone','Skin']  # 인덱스 설정
 
# NIR-1 적외선의 감쇠계수 합 
ac_nir1 = attenuation_coefficient.loc[:,'NIR-1'].sum()
# 감쇠 길이 DataFrame
attenuation_length= pd.DataFrame({'NIR-1' : [1.0,0.2,0.4], 
                                  'NIR-2' : [1.35,0.22,0.44],
                                  'SWIR' : [1.54,0.23,0.47],
                                  'SWIR-2' : [1.11,0.2,0.41]})
# 감쇠 길이 index설정 (뇌 피질, 두개골, 피부)
attenuation_length.index = ['Brain_cortex','Cranial_bone','Skin']  # 인덱스 설정
# NIR-1 적외선의 감쇠길이 합 
al_l = attenuation_length.loc[:,'NIR-1'].sum() / 10 # mm -> cm


# 흡광도 2개
def tuning_var(s):
    s_rho = s['rho'] / 10                        # _rho (mm -> cm)
    s_src = s[s.index.str.endswith('src')]       # _src
    s_dst = s[s.index.str.endswith('dst')]       # _dst    

    # index 표준화
    set_index = s_src.index.str.split('_').str[0]
    s_src.index = set_index
    s_dst.index = set_index

    #lambert beer 법칙
    # T(투광도) = I(투과방사선)/I0(입사방사선)
    # A(흡광도) = -log(T)  
    #           = ε(흡광계수) ⋅ b(투과 경로 길이(cm)) ⋅ c(몰농도)
    #           = 2 - log(%T) ***
    #           = log(1/T)
    
    # 투광도 = I(투과방사선)/I0(입사방사선)
    transmittance = 1/(s_dst/s_src)
    
    # 계산 완료후 inf,nan 0으로 치환
    transmittance = [i if i != np.inf else 0.0 for i in transmittance ]
    transmittance = Series(transmittance).fillna(value = 0)

    # math.log 계산을 위해 0을 1로 치환후 계산(흡광계수는 1로 가정한다.)
    transmittance = Series([1 if i == 0 else i for i in transmittance ])
    
    #흡광도_1 : -log10(I(투과방사선)/I0(입사방사선))
    absorbance_1 = Series(map(lambda x : (math.log(x,10)),transmittance))
    
    #흡광도_2 :  ε(흡광계수) ⋅ b(투과 경로 길이(cm)) ⋅ c(농도) (농도는 1로 가정) 
    # 흡광계수는 감쇠계수 * 감쇠길이(cm)로 사용
    c = 1
    final_nir1 = al_l * ac_nir1
    absorbance_2 = Series((s_rho * final_nir1 * c))
    
    # 흡광도 index 설정
    absorbance_1.index = set_index.map(lambda x : 'A1_' + x)
    absorbance_2.index = ['A2_rho']
    
    # 두 Series의 병합
    out_s = Series()
    out_s = out_s.append(absorbance_2).append(absorbance_1)
    
    return(out_s)




tuning_x_train =  x_train_hhb.apply(tuning_var, axis=1) # 7500 x 35
tuning_x_test = x_test_hhb.apply(tuning_var, axis =1)   # 2500 x 35

tuning_x_train = tuning_x_train.astype('float64')
tuning_x_test = tuning_x_test.astype('float64')

scaler = StandardScaler()
scaler.fit(tuning_x_train, y_train_hhb)

x_scaled_hhb = scaler.transform(tuning_x_train)
test_x_scaled_hhb = scaler.transform(tuning_x_test)


# RandomForestRegressor
m_rf = rf_r()
model = m_rf.fit(tuning_x_train,y_train_hhb.values.ravel())

m_rf.score(tuning_x_train,y_train_hhb)   # 0.98
m_rf.score(tuning_x_test,y_test_hhb.values.ravel())     # 0.85
# min_samples_leaf=1, min_samples_split=2,
# n_estimators=10,max_features='auto'




# 매개변수 튜닝
-----------------------------------------------------------------
# n_estimators = 3 
score_train=[]; score_test=[]
for i in np.arange(1,101):
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
m_rf = rf_r(n_estimators=41, min_samples_split = 31, max_features = 10, max_depth = 30)
m_rf.fit(x_scaled_hhb, y_train_hhb)
m_rf.score(x_scaled_hhb,y_train_hhb.values.ravel())   
m_rf.score(test_x_scaled_hhb, y_test_hhb.values.ravel())    


-------------------------basic------------------------------

# 랜덤하이퍼 파라미터 그리드
# RandomizedSearchCV를 사용하려면 먼저 피팅하는 동안 샘플링 할 매개 변수 그리드를 만들어야합니다.

# 현재 포리스트
pprint (rf_r.get_params(m_rf))

# 임의 포리스트의 트리 수
n_estimators = [int (x) for x in np.linspace (start = 200, stop = 2000, num = 10)]

# 분할 할 때마다 고려해야 할 기능 수 
max_features = ['auto','aqrt']

# 트리의 최대 레벨 수
max_depth = [int(x) for x in np.linspace (10, 110, num = 11)]
max_depth.append(None)


min_samples_split = [2,5,10]
min_samples_leaf = [1,2,4]
bootstrap = [True, False]


# random grid 생성
random_grid = {'n_estimators':n_estimators,
               'max_features':max_features,
               'max_depth':max_depth,
               'min_samples_split':min_samples_split,
               'max_features':max_features,
               'bootstrap':bootstrap}

pprint(random_grid)

# 

from sklearn.ensemble import RandomForestRegressor 
rf = RandomForestRegressor(random_state = 42)

rf_random = RandomizedSearchCV(estimator = rf_r, param_distributions = random_grid,
                                 n_iter=100, cv = 3, verbose = 2, 
                                 random_state = 42, n_jobs = -1,
                                 scoring = 'neg_mean_absolute_error')

rf_random.evaluate(test_x_scaled_hhb, y_test_hhb)


# error Cannot clone object '<class 'sklearn.ensemble._forest.RandomForestRegressor'>'\
# (type <class 'abc.ABCMeta'>): 
# it does not seem to be a scikit-learn estimator as it does not implement a 'get_params' methods.


--------------------------------------------------
x_scaled_hhb
test_x_scaled_hhb

# 평가방법 튜닝

def evaluate(model, test_x_scaled_hhb, y_test_hhb):
    predictions = model.predict(test_x_scaled_hhb)
    errors = abs(predictions - y_test_hhb)
    mape = 100 * np.mean(errors / y_test_hhb)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    return accuracy

base_model = RandomForestRegressor(n_estimators = 50, random_state = 42, max_depth=30)
base_model.fit(x_scaled_hhb, y_train_hhb.values.ravel())
base_accuracy1 = evaluate(base_model, test_x_scaled_hhb, y_test_hhb.values.ravel())

base_accuracy = evaluate(base_model, x_scaled_hhb, y_train_hhb.values.ravel())

base_accuracy.dtype



model = Sequential()
model.add(Dense(18, input_dim=36, activation='relu')) 
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['MAE'])


base_model.fit(x_scaled_hhb,y_train_hhb)


def model_x(x_scaled_hhb, y_train_hhb, number):
   
    # 모델의 설정
    model = Sequential() 
    model.add(Dense(18, input_dim=36, activation='relu')) 
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='relu'))
    
    # 모델 컴파일
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['MAE'])
 
    # 모델 실행 
    model.fit(x_scaled_hhb, y_train_hhb, epochs=number, batch_size=10) 
    return(model)


model_hhb = model_x(x_scaled_hhb,y_train_hhb,500)
model_hhb.evaluate(x_scaled_hhb,y_train_hhb)[1]

# 0.75


hhb.dtype('float64')

from keras.models import load_model
model_hhb.save('base_accuracy_h5')

dir(model)
model2_hhb = load_model('base_accuracy_h5')
test1 = pd.read_csv('sample_submission.csv',index_col='id')
test1 = test1.astype('float')

tunning_realtest_x = test.apply(tuning_var,axis=1)
realtest_x_scaled = scaler.transform(tunning_realtest_x)
test1.hhb = model2_hhb.predict(realtest_x_scaled)


test1.iloc[:,0] = test1.hhb
test1.hhb.to_csv('hhb.csv')

DataFrame(model_hhb)





tunning_realtest_x = test.apply(tuning_var, axis=1)
realtest_x_scaled = scaler.transform(tunning_realtest_x)

test1.hhb = pd.read


# 범위 좁히기 GridSearchCV

from sklearn.model_selection import GridSearchCV

param_grid = {
    'bootstrap':[True],
    'max_depth':[80,90,100,110],
    'max_features':[2,3],
    'min_samples_leaf':[3,4,5],
    'min_samples_split':[8,10,12],
    'n_estimators':[100,200,300,1000]
    }

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                           cv=3, n_jobs=-1, verbose=2)

grid_search.fit(tuning_x_train, y_train_hhb.values.ravel())
grid_search.score(tuning_x_test, y_test_hhb.values.ravel())



