import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import OneHotEncoder
from src.config import config


class Prophet:
    def __init__(self, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, holidays=None):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays = holidays
        self.params = {}

    def fit(self, df):
        self.df = df.copy()
        self.df['t'] = (self.df['ds'] - self.df['ds'].min()) / np.timedelta64(1, 'D')
        self.y = self.df['y'].values

        # Prepare seasonal features
        seasonal_features = []
        if self.yearly_seasonality:
            seasonal_features.append(self.fourier_series(self.df['ds'], 365.25, 10))
        if self.weekly_seasonality:
            seasonal_features.append(self.fourier_series(self.df['ds'], 7, 3))
        if self.daily_seasonality:
            seasonal_features.append(self.fourier_series(self.df['ds'], 1, 3))
        if self.holidays is not None:
            holidays_enc = OneHotEncoder(sparse=False).fit_transform(self.df[['holiday']])
            seasonal_features.append(holidays_enc)

        self.seasonal_features = np.hstack(seasonal_features)

        # Initialize parameters
        self.params['beta'] = np.zeros(self.seasonal_features.shape[1])
        self.params['k'] = 0
        self.params['m'] = 0

        # Optimize parameters
        initial_params = np.concatenate([self.params['beta'], [self.params['k'], self.params['m']]])
        res = minimize(self.loss, initial_params, method='L-BFGS-B')
        self.params['beta'] = res.x[:-2]
        self.params['k'] = res.x[-2]
        self.params['m'] = res.x[-1]

    def fourier_series(self, dates, period, series_order):
        t = (dates - dates.min()) / np.timedelta64(1, 'D')
        return np.hstack([
                             np.sin((2.0 * (i + 1) * np.pi * t / period))[:, np.newaxis] for i in range(series_order)
                         ] + [
                             np.cos((2.0 * (i + 1) * np.pi * t / period))[:, np.newaxis] for i in range(series_order)
                         ])

    def loss(self, params):
        beta = params[:-2]
        k = params[-2]
        m = params[-1]
        y_hat = k * self.df['t'] + m + np.dot(self.seasonal_features, beta)
        return np.mean((self.y - y_hat) ** 2)

    def predict(self, future):
        future['t'] = (future['ds'] - self.df['ds'].min()) / np.timedelta64(1, 'D')

        # Prepare seasonal features
        seasonal_features = []
        if self.yearly_seasonality:
            seasonal_features.append(self.fourier_series(future['ds'], 365.25, 10))
        if self.weekly_seasonality:
            seasonal_features.append(self.fourier_series(future['ds'], 7, 3))
        if self.daily_seasonality:
            seasonal_features.append(self.fourier_series(future['ds'], 1, 3))
        if self.holidays is not None:
            holidays_enc = OneHotEncoder(sparse=False).fit_transform(future[['holiday']])
            seasonal_features.append(holidays_enc)

        seasonal_features = np.hstack(seasonal_features)
        future['yhat'] = self.params['k'] * future['t'] + self.params['m'] + np.dot(seasonal_features,
                                                                                    self.params['beta'])
        return future[['ds', 'yhat']]