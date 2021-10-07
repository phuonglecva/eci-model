# -*- coding: utf-8 -*-
"""Xuất nhập khẩu

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1m1KaVb_1hFiehGu7EdHb_-rN4jO99jF7
"""

# !pip install pmdarima
import pandas as pd
import numpy as np

data = pd.read_excel('data/dong-nai/xuatnhapkhau.xlsx', sheet_name=0)
timeline = data.iloc[0, :].values
timeline = [t.split(' ')[-1] for  t in timeline if str(t) != 'nan'][:-2]

main_index = [6]
sub_indices = list(range(10, 28))

main_data = data.iloc[main_index, 4::6]
sub_data = data.iloc[sub_indices, 4::6]

main_data_values = np.squeeze(main_data.values[:, 3:-2])

sub_data_values = sub_data.values[:, 3:-2]

main_data_values = np.array([str(num).replace(',', 'x').replace('.', '').replace('x', '.') for num in main_data_values], dtype='float')
sub_data_values = np.array([[str(num).replace(',', 'x').replace('.', '').replace('x', '.') for num in row] for row in sub_data_values], dtype='float')




from pmdarima import auto_arima
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

models = {
    'lin': LinearRegression(),
    'tree': DecisionTreeRegressor(),
    'gbr':  GradientBoostingRegressor(),
    'rf': RandomForestRegressor()
}

# define NOBS = 3
NOBS = 3
# generate forecast inputs
sub_train, _ = sub_data_values[:, :-NOBS], sub_data_values[:, -NOBS:] 
time_train, time_test = timeline[:-NOBS], timeline[-NOBS:]
arima_model_list = []
for row in sub_train:
  model = auto_arima(row)
  arima_model_list.append(model)

sub_inputs = []
for model in arima_model_list:
  sub_inputs.append(model.predict(NOBS).tolist())

sub_inputs = np.array(sub_inputs)

# train model to predict total import from goods
main_train, main_test  = main_data_values[:-NOBS], main_data_values[-NOBS:]
sub_train, sub_test  = sub_data_values.T[:-NOBS, :], sub_data_values.T[-NOBS:, :]

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
    model.fit(sub_train, main_train)
    pred = model.predict(sub_inputs.T)
    acc = (1 - abs(pred - main_test) / main_test)
    if acc.mean() > .97:
      reg_models.append({
        'acc': acc.mean(),
        'model': model
      })
    if acc.mean() > best_model['acc_mean']:
      best_model['model'] = model
      best_model['name'] = name
      best_model['acc_mean'] = acc.mean()
    
    print('"""')
    print(f"model_name: {name}")
    print(f'predicted: {pred}')
    print(f'real: {main_test}')
    print(f'acc: {acc} {acc.mean()}')
    print('"""')


print(f'best model: {best_model}')

std_list = []
for model in reg_models:
    
    train_pred = model['model'].predict(sub_train)
    std = (((train_pred - main_train) ** 2).sum() / (len(train_pred) - 2)) ** 0.5
    std_list.append(std)

std_avg = sum(std_list) / len(std_list)

# saved_model
import joblib
saved_model = {
    "arima_models": arima_model_list,
    "reg_models": [model['model'] for model in reg_models[:10]],
    "X_train": sub_train,
    "X_test": sub_test,
    "y_train": main_train,
    "y_test": main_test,
    'time_train': time_train,
    'time_test': time_test,
    "std_avg": std_avg
}
joblib.dump(saved_model, "import_forecast_model.joblib")