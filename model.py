from numpy.core.fromnumeric import compress
from data_loader import convert_low2high, get_gdp_data, load_val_from_ggtrends
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from pmdarima import auto_arima
import os
import joblib


class ForecastingRegression:
    def __init__(self, factors, target):
        self.target = target
        self.factors = factors
        self.model = GradientBoostingRegressor()
        self.ggtrends = load_val_from_ggtrends()
        gdp_years, _, gdp_rate = get_gdp_data()
        gdp_years, gdp_data_m = convert_low2high(gdp_years, gdp_rate)

        gdp_data = gdp_data_m[7] + gdp_data_m[8] + \
            gdp_data_m[9] + gdp_data_m[10][:7]

        gdp_arr = np.array(gdp_data)
        self.gdp_arr = gdp_arr.reshape((gdp_arr.shape[0], -1))
        self.factors = np.concatenate([factors, self.gdp_arr, self.ggtrends], axis=1)
        self.arimas_list = []

    def split_test(self, NOBS=3):
        self.X_train, self.X_test = self.factors[: -NOBS], self.factors[-NOBS:]
        self.y_train, self.y_test = self.target[:-NOBS], self.target[-NOBS:]  

    def fit(self):
        params = {'n_estimators': 2000,
                  'max_depth': 3,
                  'min_samples_split': 3,
                  'learning_rate': 0.001,
                  'loss': 'ls'}
        self.model = GradientBoostingRegressor(**params)
        self.model.fit(self.X_train, self.y_train)

    def predict(self, next_month=3):
        inputs = self.generate_next_n_inputs(next_months=next_month)
        return self.model.predict(inputs)
    
    def fit_arimas(self):
        for i in range(len(self.X_train.T)):
            train_data, _ = self.X_train.T[i], self.X_test.T[i]
            model = auto_arima(train_data)
            self.arimas_list.append(model)

    
    def generate_next_n_inputs(self, NOBS=3, next_months=3):
        """
        Parameters explaination:
        - NOBS: number of month use to  test
        - next_month: num of next month forecasted by auto_arima 
        """
        X_train = self.X_train
        X_test = self.X_test
        sub_forecasts = []
        for i in range(len(X_train.T)):
            index = i
            train_data, _ = X_train.T[index], X_test.T[index]
            model = self.arimas_list[i]
            test_hat = model.predict(next_months)
            sub_forecasts.append(test_hat.tolist())
            self.arimas_list.append(model)

        sub_input = np.array(sub_forecasts).T
        return sub_input

    def save(self, base_path='checkpoints/'):
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        joblib.dump(self.factors, base_path + 'factors.joblib', compress=True)
        joblib.dump(self.target, base_path + 'target.joblib', compress=True)
        joblib.dump(self.model, base_path + 'model.joblib', compress=True)
        joblib.dump(self.arimas_list, base_path + 'arimas_list.joblib', compress=True)
    
    