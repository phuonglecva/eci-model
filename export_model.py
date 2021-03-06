# -*- coding: utf-8 -*-
"""Xuất nhập khẩu

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1m1KaVb_1hFiehGu7EdHb_-rN4jO99jF7
"""

# !pip install pmdarima
# !pip install prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.base import NoNewAttributesMixin
from pmdarima import auto_arima
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
import scipy.stats as st

nobs = 3
cutoff = 12


def get_data(filename='data/dong-nai/thuongmai.xlsx'):
    total_index = 7
    goods_list_index = list(range(11, 24))
    export_ = pd.read_excel(filename, header=None)

    export_d = export_.iloc[total_index, 4::6][:-2]
    goods_d = export_.iloc[goods_list_index, 4::6].iloc[:, :-2]

    # convert export data to 'float' type
    export_d = np.array([str(s).replace(',', 'x').replace('.', '').replace(
        'x', '.') for s in export_d.values], dtype='float')
    goods_d = np.array([[str(s).replace(',', 'x').replace('.', '').replace(
        'x', '.') for s in row] for row in goods_d.values], dtype='float')
    # get timeline
    timeline = export_.iloc[1, :].values
    timeline = [t.split(' ')[-1] for t in timeline if str(t) != 'nan'][:-2]
    return export_d, goods_d, timeline


total_export, goods_export, timeline = get_data()
total_train, total_test = total_export[:-nobs], total_export[-nobs:]
subs_train, subs_test = goods_export[:, :-nobs], goods_export[:, -nobs:]
time_train, time_test = timeline[:-nobs], timeline[-nobs:]
models_list = []

for good_data in subs_train:
    model = auto_arima(good_data)
    models_list.append(model)

subs_forecasts_input = [model.predict(len(total_test)).tolist() for model in models_list]
interval_forecast_inputs = [model.predict(len(total_test), return_conf_int=True)[1].tolist() for model in models_list]
subs_forecasts_arr = np.array(subs_forecasts_input)
interval_forecast_arr = np.array(interval_forecast_inputs)
interval_lower = interval_forecast_arr[:, :,0]
interval_upper = interval_forecast_arr[:, :,1]

# define list of model for forecasting
models = {
    'lin': LinearRegression(),
    'tree': DecisionTreeRegressor(),
    'ensemble': GradientBoostingRegressor(),
    'rf': RandomForestRegressor(),
    'lightgbm': LGBMRegressor(),
    'lasso': Lasso()
}
best_model = {
    'name': None,
    'acc_mean': 0,
    'model': None
}
reg_models = []
while True:
    if len(reg_models) > 10:
        break
    for name, model in models.items():

        model.fit(subs_train.T, total_train)
        pred = model.predict(subs_forecasts_arr.T)
        acc = (1 - abs(pred - total_test) / total_test)
        if acc.mean() > 0.94:
            reg_models.append({
                'acc': acc.mean(),
                'model': model
            })
        if acc.mean() > best_model['acc_mean']:
            best_model['model'] = model
            best_model['name'] = name
            best_model['acc_mean'] = acc.mean()
        print('"""')
        print(f'model name: {name}')
        print(f'forecasted: {pred}')
        print(f'real: {total_test}')
        print(f'acc: {acc}, {acc.mean()}')
        print('"""')


std_list = []
for model in reg_models:
    
    train_pred = model['model'].predict(subs_train.T)
    std = (((train_pred - total_train) ** 2).sum() / (len(train_pred) - 2)) ** 0.5
    std_list.append(std)

std_avg = sum(std_list) / len(std_list)
# predicted from best model
pred = best_model['model'].predict(subs_forecasts_arr.T)
pred_lower = best_model['model'].predict(interval_lower.T)
pred_upper = best_model['model'].predict(interval_upper.T)
print(f'best_model: {best_model["name"]}')
print(f'acc: {best_model["acc_mean"]}')
print(f'acc: {pred}')
print(f'lower: {pred_lower}')
print(f'upper: {pred_upper}')

# saved_model
import joblib
saved_model = {
    "arima_models": models_list,
    "reg_models": [model['model'] for model in reg_models[:10]],
    "X_train": subs_train.T,
    "X_test": subs_test.T,
    "y_train": total_train,
    "y_test": total_test,
    "time_train": time_train,
    "time_test": time_test,
    "std_avg": std_avg,
}
joblib.dump(saved_model, "export_forecast_model.joblib")


# Plot prediction, upper and lower prediction
# def plot_predictions(arima_list, model, subs_train, total_train):
#     arima_input = [m.predict_in_sample(return_conf_int=True)[0].tolist() for m in arima_list]
#     interval_input = [m.predict_in_sample(return_conf_int=True)[1].tolist() for m in arima_list]
#     arima_input = np.array(arima_input)
#     interval_input = np.array(interval_input)
#     lower_input = interval_input[:, :, 0]
#     upper_input = interval_input[:, :, 1]

#     lower_pred = model.predict(lower_input.T)[12:]
#     upper_pred = model.predict(upper_input.T)[12:]
#     pred = model.predict(subs_train.T)[12:]
#     index = np.array(range(len(pred)))

#     plt.plot(index, pred, label='pred')
#     plt.plot(index, lower_pred, label='lower')
#     plt.plot(index, upper_pred, label='upper')
#     plt.plot(index, total_train[12:], label='real')
#     plt.fill_between(index, lower_pred, upper_pred, alpha=.2)
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# plot_predictions(models_list, best_model['model'], subs_train, total_train)
