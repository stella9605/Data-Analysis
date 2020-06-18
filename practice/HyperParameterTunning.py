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

# hbo2
x_train_hbo2, x_test_hbo2, y_train_hbo2, y_test_hbo2 = train_test_split(X,
                                                                        Y_2,
                                                                        random_state = 0)

# ca 
x_train_ca, x_test_ca, y_train_ca, y_test_ca = train_test_split(X,
                                                                Y_3,
                                                                random_state = 0)

# na
x_train_na, x_test_na, y_train_na, y_test_na = train_test_split(X,
                                                                Y_4,
                                                                random_state=0)



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

tuning_x_train_hhb = x_train_hhb.apply(tuning_var, axis=1)
tuning_x_test_hhb = x_test_hhb.apply(tuning_var, axis=1)

# 튜닝 변수 스케일링
m_sacled = StandardScaler()
m_sacled.fit(tuning_x_train_hhb)

x_scaled_hbb = m_sacled.transform(tuning_x_train_hhb)
test_x_scaled_hbb = m_sacled.transform(tuning_x_test_hhb)



#RandomForestRegressor, No Scaling, for hhb 
from sklearn.ensemble import RandomForestRegressor
rf_r = RandomForestRegressor(random_state = 42)

from pprint import pprint

Parameters currently in use:
    {'bootstrap': True,
     'criterion': 'mse',
     'max_depth': None,
     'max_features': 'auto',
     'max_leaf_nodes': None,
     'min_impurity_decrease': 0.0,
     'min_impurity_split': None,
     'min_samples_leaf': 1,
     'min_samples_split': 2,
     'min_weight_fraction_leaf': 0.0,
     'n_estimators': 10,
     'n_jobs': 1,
     'oob_score': False,
     'random_state': 42,
     'verbose': 0,
     'warm_start': False}

print('Parameters currently in use:\n')
pprint(rf_r.get_params())




m_rf_hhb = rf_r
m_rf_hhb.fit(tuning_x_train_hhb,y_train_hhb)
m_rf_hhb.score(tuning_x_test_hhb, y_test_hhb)








# user the forest's method on the test data 
y_train_predict = m_rf_hhb.predict(tuning_x_train_hhb)
y_test_predict = m_rf_hhb.predict(tuning_x_test_hhb)

mean_absolute_error(y_train_hhb, y_train_predict)      # 0.42
mean_absolute_error(y_test_hhb, y_test_predict)        # 1.12


# Random Hyperparameter Gride
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators':n_estimators,
                'max_features':max_features,
                'max_depth':max_depth,
                'min_samples_split':min_samples_split,
                'min_samples_leaf':min_samples_leaf,
                'bootstrap':bootstrap}

pprint(random_grid)

m_rf_hhb.fit(tuning_x_train_hhb,y_train_hhb)
m_rf_hhb.score(tuning_x_test_hhb, y_test_hhb)

predict = m_rf_hhb.fit(tuning_x_train_hhb,y_train_hhb)







# grid hyper parameter
n_estimators = [10,20,30]
max_features = [2,4]
bootstrap = [False]


rf_random = RandomizedSearchCV(estimator = rf_r,
                               param_distributions  = random_grid,
                               n_iter = 100,
                               cv = 3,
                               verbose = 2,
                               random_state = 42,
                               n_jobs = 2)

x_train_hhb = x_train_hhb.astype(float)
y_train_hhb = y_train_hhb.astype(float)

rf_random.fit(x_train_hhb, y_train_hhb)
rf_random.fit(tuning_x_train_hhb, y_train_hhb)

rf_random.best_params_








param_grid = [{
    'n_estimators':n_estimators, 'max_features':max_features},
    {'bootstrap':bootstrap,'n_estimators':n_estimators, 'max_features':max_features }
    ]


grid_search = GridSearchCV(gb_r, param_grid = param_grid,
                           cv=2, scoring='mean_absolute_error',
                           verbose=2,
                           n_jobs=-1,
                           return_train_score=True)

grid_search.fit(tuning_x_train_hhb, y_train)


# Label Encoding
lab_enc = preprocessing.LabelEncoder()
y_train_hhb = lab_enc.fit_transform(y_train_hhb)






# DT
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=99)
tree.fit(tuning_x_train_hbb, y_train)

print("훈련 세트 정확도: {:.3f}".format(tree.score(tuning_x_train_hbb, y_train)))
print("테스트 세트 정확도: {:.3f}".format(tree.score(tuning_x_test_hbb, y_test)))






scaler = StandardScaler()
tuning_x_train_scaler = scaler.fit_transform(tuning_x_train_hbb)
tuning_x_test_scaler = scaler.transform(tuning_x_test_hbb)

from keras.models import Sequential 
from keras.layers.core import Dense 


model = Sequential()
model.add(Dense(36, input_dim = tuning_x_train_scaler.shape[1], activation = 'relu'))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mean_absolute_error',
              optimizer='adam',
              metrics=['MAE']) 

model.fit(tuning_x_train_scaler, y_train_hbb, epochs = 100, batch_size = 100)
model.evaluate(tuning_x_test_scaler, y_test_hbb)[1]  # 2.06

















































