import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

DATA_DIR = 'data/dong-nai'


def get_cpi_data(filename='{}/cpi_data.csv'.format(DATA_DIR)):
    data = pd.read_csv(filename, encoding='latin', header=None)
    # X_data, y_data = data.values[1:, 1:].T, data.values[:1, 1:].T
    values = data.values
    dataX, dataY = np.array(values[1:].T, dtype='float'), np.array(values[0], dtype='float')
    dataX = dataX - 100
    dataY = dataY - 100
    return dataX, dataY


def get_gdp_data(filename='{}/gdp.xlsx'.format(DATA_DIR)):
    data = pd.read_excel(filename)
    values = data.values
    year = values[1:, 0]
    val = values[1:, 1]
    rate = values[1:, 2]

    year = np.array(year, dtype='int')
    val = np.array([v.replace('.', 'x').replace(
        ',', '.').replace('x', '') for v in val], dtype='float')
    rate = np.array([r.replace(',', '.') for r in rate], dtype='float')
    return year, val, rate

def get_unemployment_data(filename='{}/thatnghiep.xlsx'.format(DATA_DIR)):
    data = pd.read_excel(filename)

    return data

def convert_low2high(index, data):
    min_err = 1000
    min_deg = 1000
    for deg in range(len(data)):
        # print(deg)
        f_map = np.polyfit(index, data, deg)
        f = np.poly1d(f_map)
        pred = np.array([f(i) for i in index], dtype='float')
        err = mean_squared_error(pred, data)
        if err < min_err:
            min_err = err
            min_deg = deg

    f_map = np.polyfit(index, data, min_deg)
    f = np.poly1d(f_map)

    return index, [[f(year - (1 / 12) * i) for i in reversed(range(12))] for year in index]


def load_val_from_ggtrends(filename='ggtrends/'):
    # name = ['thunhap.csv', 'congviec.csv', 'kcn.csv', 'vieclam.csv']
    name = ['congviec.csv', 'kcn.csv', 'vieclam.csv']
    res = []
    for n in name:
        df = pd.read_csv(filename + n)
        df['date'] = pd.to_datetime(df['time'],format='%d/%m/%Y')
        df['year'] = pd.DatetimeIndex(df['date']).year
        df['month'] = pd.DatetimeIndex(df['date']).month
        val = df.groupby(['year', 'month']).sum()
        res.append({
            'name': n,
            'val': val.values[:, 0].tolist()
        })
    return np.array([row['val'] for row in res]).T