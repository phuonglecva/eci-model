from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

def select_arima_model(series, nobs=3):
  train, test = series[:-nobs], series[-nobs:]
  min = 1000
  for i in range(3):
    for j in range(3):
      for k in range(3):
        try:
          model = ARIMA(train, order=(i, j, k))
          model_fitted = model.fit()
          test_hat = model_fitted.forecast(nobs)[0]
          err = mean_squared_error(test_hat, test, squared=False)
          if err < min:
            best_model = model_fitted
        except:
          continue
  return best_model