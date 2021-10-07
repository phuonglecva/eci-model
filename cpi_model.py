import sys
from data_loader import *
from side_model import *
from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNet, Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from pmdarima import auto_arima
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
import joblib
from datetime import datetime
from mapie.regression import MapieRegressor
from pmdarima.arima.utils import ndiffs

sub_cpi_data, main_cpi_data = get_cpi_data()
gdp_years, gdp_value, gdp_rate = get_gdp_data()
gdp_years, gdp_data_m = convert_low2high(gdp_years, gdp_rate)

gdp_data = gdp_data_m[7] + gdp_data_m[8] + gdp_data_m[9] + gdp_data_m[10][:7]

gdp_arr = np.array(gdp_data)
gdp_arr = gdp_arr.reshape((gdp_arr.shape[0], -1))

ggtrends_data = load_val_from_ggtrends()

gdp_arr2 = gdp_arr ** 2
X_data = np.concatenate([sub_cpi_data, gdp_arr, ggtrends_data], axis=1)
# X_data = sub_cpi_data
y_data = main_cpi_data

NOBS = 3
# get timeline
timeline = pd.read_excel('data/dong-nai/cpi_timeline.xlsx').columns
timeline = [datetime.strftime(t, '%m/%y') for t in timeline.tolist()]
time_train, time_test = timeline[:-NOBS], timeline[-NOBS:]


def train_test_split(X, y, nobs=3):
    trainX, testX = X[: -nobs], X[-nobs:]
    trainY, testY = y[:-nobs], y[-nobs:]
    return trainX, testX, trainY, testY


X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, NOBS)

sub_forecasts = []
arima_models = []
for i in range(len(X_train.T)):
    index = i
    train_data, test_data = X_train.T[index], X_test.T[index]
    print("=========================================")
    n_d = ndiffs(train_data, test='kpss')
    # model = auto_arima(train_data, alpha=0.01, d = n_d)
    model = auto_arima(train_data, alpha=0.01)
    # print("N-diff: ", n_d)
    # print("Data: ", train_data[:5])
    # print("Params: ", model.get_params()['order'])
    print("=========================================")
    arima_models.append(model)
    test_hat = model.predict(NOBS)
    sub_forecasts.append(test_hat.tolist())
sub_input = np.array(sub_forecasts).T


# exit(1)
# params = {'n_estimators': 2000,
#           'max_depth': 3,
#           'min_samples_split': 3,
#           'learning_rate': 0.001,
#           'loss': 'ls'}
# for lin in [GradientBoostingRegressor(**params) for i in range(10)]:
#     lin.fit(X_train, y_train)
#     print("sub forecasting")
#     lin_hat = lin.predict(sub_input)
#     acc = (1  - abs(lin_hat - y_test) / (y_test - 100))
#     print(lin_hat, '\n', y_test)

#     print(f'acc: {acc} {acc.mean()}')
#     print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')

# params = {'n_estimators': [1000*i for i in range(4)],
#           'max_depth': [1, 3, 5, 7],
#           'min_samples_split': [1, 3, 5, 7],
#           'learning_rate': [.1, .03, .01, .003, .001],
#           'loss': ['ls', 'huber']}
# model = GradientBoostingRegressor()
# tunning_model = GridSearchCV(model, param_grid=params,
#                              scoring='neg_mean_squared_error', cv=3, verbose=3, n_jobs=-1)
# tunning_model.fit(X_train, y_train)

# print('""""')
# print(f'best params: {tunning_model.best_params_}')
# model = GradientBoostingRegressor(**tunning_model.best_params_)
# model.fit(X_train, y_train)
# pred = model.predict(sub_input)
# acc = (1 - abs(pred - y_test) / (y_test - 100))
# print('test:', y_test)
# print('predicted:', pred)
# print(f'acc: {acc}, {acc.mean()}')
# print('""""')

# define list of model for forecasting
# models = {
#     'lin': LinearRegression(),
#     'tree': DecisionTreeRegressor(),
#     'ensemble': GradientBoostingRegressor(),
#     'rf': RandomForestRegressor(),
#     'xgb': XGBRegressor(),
#     'lasso1': Lasso(alpha=1),
#     'elastic': ElasticNet(),
#     'lgbm': LGBMRegressor(),
#     'ridge': Ridge(),
# }
# best_model = {
#     'name': None,
#     'acc_mean': 0,
#     'model': None
# }
# best_acc = 0
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     pred = model.predict(sub_input)
#     acc = (1 - abs(pred - y_test) / (y_test))
#     if acc.mean() > best_model['acc_mean']:
#         best_model['model'] = model
#         best_model['name'] = name
#         best_model['acc_mean'] = acc.mean()
#     print('"""')
#     print(f'model name: {name}')
#     print(f'forecasted: {pred}')
#     print(f'real: {y_test}')
#     print(f'acc: {acc}, {acc.mean()}')
#     print(f'params: {model.get_params()}')
#     print('"""')

# print(f'best_model: {best_model["name"]}')
# print(f'acc: {best_model["acc_mean"]}')

# params = {'alpha': 0.9, 'ccp_alpha': 0.0, 'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.1, 'loss': 'ls', 'max_depth': 3, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_iter_no_change': None, 'random_state': None, 'subsample': 1.0, 'tol':
#           0.0001, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}
# reg_models = []
# while True:
#     if len(reg_models) == 10:
#         break
#     model = GradientBoostingRegressor(**params)
#     # model  = GradientBoostingRegressor()
#     mapie = MapieRegressor(model)
#     mapie.fit(X_train, y_train)

#     # pred = model.predict(sub_input)
#     pred, intervals = mapie.predict(sub_input, alpha=0.1)
#     acc = (1 - abs(pred - y_test) / (y_test))
#     if acc.mean() > 0.90 and round(pred[1], 2) != round(pred[2], 2):
#         reg_models.append({
#             'acc': acc.mean(),
#             # 'model': model
#             'model': mapie
#         })
#         print('"""')
#         print(f'forecasted: {pred}')
#         print(f'real: {y_test}')
#         print(f'acc: {acc}, {acc.mean()}')
#         print(f'intervals: {intervals}')
#         print('""""')

# std_list = []
# for model in reg_models:

#     train_pred, _ = model['model'].predict(X_train, alpha=.1)
#     std = (((train_pred - y_train) ** 2).sum() / (len(train_pred) - 2)) ** 0.5
#     std_list.append(std)

# std_avg = sum(std_list) / len(std_list)


# saved_model
# saved_model = {
#     "arima_models": arima_models,
#     "reg_models": [model['model'] for model in reg_models],
#     "X_train": X_train,
#     "X_test": X_test,
#     "y_train": y_train,
#     "y_test": y_test,
#     "time_train": time_train,
#     "time_test": time_test,
#     "std_avg": std_avg
# }
# joblib.dump(saved_model, "cpi_forecast_model.joblib")
